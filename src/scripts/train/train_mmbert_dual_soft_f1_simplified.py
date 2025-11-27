"""
Dual-head multi-label PyTorch training script for mmBERT-base.
Two classification heads: onderwerp (topic) and beleving (experience) with dynamic label counts.
Uses combined F1+BCE loss with weight α (configurable balance).
Features: learnable thresholds, warmup + cosine LR, gradient clipping.
mmBERT: Modern multilingual encoder (1800+ languages, 2x faster than XLM-R).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer, AutoModel
import os
import json
import numpy as np
import argparse
import random
import wandb
import bitsandbytes as bnb
from dataset_loader import load_wim_dataset


# Threshold helpers: logit ↔ probability conversions
def prob_to_logit(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert probabilities to logits (inverse sigmoid). Numerically stable."""
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))


def logit_to_prob(l: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities using sigmoid."""
    return torch.sigmoid(l)


# Set device - MPS for Apple Silicon, fallback to CPU
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def set_seed(seed):
    """Set random seeds for reproducibility across torch, numpy, and Python random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class mmBERTDualHead(nn.Module):
    """
    mmBERT with two classification heads for multi-task learning.
    Shared encoder with separate heads for onderwerp and beleving.
    Optionally includes learnable thresholds for each head.
    """
    def __init__(self, model_name, num_onderwerp, num_beleving, dropout, initial_threshold, use_thresholds: bool = True):
        super().__init__()
        self.use_thresholds = use_thresholds

        # Shared mmBERT encoder (22 layers, 768 hidden, supports up to 8192 tokens)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for mmBERT-base

        # Classification head for onderwerp (topics)
        self.onderwerp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_onderwerp)
        )

        # Classification head for beleving (experiences)
        self.beleving_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_beleving)
        )

        # Thresholds are optionally parameterized in **logit space** (tau_logit).
        # Why: (1) avoids prob clamping and keeps grads healthy, (2) matches the space of logits,
        # (3) lets Soft-F1 express per-class decision boundaries independent of BCE calibration.
        self.onderwerp_tau_logit = None
        self.beleving_tau_logit = None
        if self.use_thresholds:
            init_logit = prob_to_logit(torch.tensor(initial_threshold))
            self.onderwerp_tau_logit = nn.Parameter(torch.full((num_onderwerp,), init_logit))
            self.beleving_tau_logit = nn.Parameter(torch.full((num_beleving,), init_logit))

    def forward(self, input_ids, attention_mask):
        # Get shared representation from mmBERT encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # mmBERT doesn't have pooler_output, use CLS token from last_hidden_state
        # Extract [CLS] token representation (first token in sequence)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Generate predictions from both heads
        onderwerp_logits = self.onderwerp_head(pooled_output)
        beleving_logits = self.beleving_head(pooled_output)

        return onderwerp_logits, beleving_logits


class DutchDualLabelDataset(Dataset):
    """Dataset for dual-label classification (onderwerp + beleving)."""

    def __init__(self, texts, onderwerp_labels, beleving_labels, tokenizer, max_length):
        self.texts = texts
        self.onderwerp_labels = onderwerp_labels
        self.beleving_labels = beleving_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'onderwerp_labels': torch.tensor(self.onderwerp_labels[idx], dtype=torch.float),
            'beleving_labels': torch.tensor(self.beleving_labels[idx], dtype=torch.float)
        }


def calculate_soft_f1(logits, labels, logit_threshold=None, temperature=1.0):
    """
    Calculate differentiable F1 score using sigmoid approximation.
    If logit_threshold is None: y_soft = sigmoid(logits * T)
    Else:                       y_soft = sigmoid((logits - logit_threshold) * T)
    Rationale:
      - With thresholds ON, Soft-F1 learns per-class decision boundaries in logit space.
      - With thresholds OFF, we follow POLA: a single, obvious source (head logits).
    Args:
        logits: Model predictions (before sigmoid)
        labels: True labels (multi-hot encoded)
        logit_threshold: Optional decision threshold in LOGIT space (None = no shift)
        temperature: Sharpness of sigmoid approximation
    Returns:
        soft_f1: Differentiable F1 score
    """
    # Compute shifted logits (or raw logits if threshold is None)
    if logit_threshold is None:
        shifted = logits * temperature
    else:
        shifted = (logits - logit_threshold) * temperature

    # Soft predictions using sigmoid
    y_pred_soft = torch.sigmoid(shifted)

    # Soft confusion matrix elements
    TP = (y_pred_soft * labels).sum(dim=-1)  # True Positives
    FP = (y_pred_soft * (1 - labels)).sum(dim=-1)  # False Positives
    FN = ((1 - y_pred_soft) * labels).sum(dim=-1)  # False Negatives

    # Differentiable F1 score
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return f1.mean()  # Average across batch


def evaluate(model, val_texts, val_onderwerp, val_beleving, tokenizer, device,
             onderwerp_names, beleving_names, num_samples, max_length, amp_dtype=torch.float16):
    """
    Evaluate model on validation set and return metrics.
    Args:
        model: The trained model
        val_texts: List of validation texts
        val_onderwerp: Validation onderwerp labels
        val_beleving: Validation beleving labels
        tokenizer: Tokenizer for encoding text
        device: Device to run evaluation on
        onderwerp_names: List of onderwerp label names
        beleving_names: List of beleving label names
        num_samples: Number of samples to evaluate (None = all)
        max_length: Max sequence length
        amp_dtype: Data type for autocasting (default: torch.float16)
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()

    # Determine number of samples to evaluate
    if num_samples is None:
        num_samples = len(val_texts)
    else:
        num_samples = min(num_samples, len(val_texts))

    # Track metrics
    onderwerp_correct = np.zeros(len(onderwerp_names))
    onderwerp_total = np.zeros(len(onderwerp_names))
    beleving_correct = np.zeros(len(beleving_names))
    beleving_total = np.zeros(len(beleving_names))

    # Track F1 components
    onderwerp_tp = 0
    onderwerp_fp = 0
    onderwerp_fn = 0
    beleving_tp = 0
    beleving_fp = 0
    beleving_fn = 0

    with torch.inference_mode():
        for i in range(num_samples):
            # Tokenize
            encoding = tokenizer(
                val_texts[i],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Get predictions with Autocast
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                onderwerp_logits, beleving_logits = model(input_ids, attention_mask)

            # Convert to probabilities
            onderwerp_probs = torch.sigmoid(onderwerp_logits.float())
            beleving_probs = torch.sigmoid(beleving_logits.float())

            # Apply learned per-class thresholds (if enabled) or fixed 0.5 cutoff
            if model.use_thresholds:
                tau_on = logit_to_prob(model.onderwerp_tau_logit)  # [C1]
                tau_be = logit_to_prob(model.beleving_tau_logit)   # [C2]
            else:
                # Fixed probability cutoff (POLA-friendly)
                tau_on = torch.full_like(onderwerp_probs[0], 0.5)
                tau_be = torch.full_like(beleving_probs[0], 0.5)

            onderwerp_pred = (onderwerp_probs > tau_on).squeeze().cpu().numpy()
            beleving_pred = (beleving_probs > tau_be).squeeze().cpu().numpy()

            # Get true labels
            onderwerp_true = val_onderwerp[i]
            beleving_true = val_beleving[i]

            # Update F1 components
            onderwerp_tp += ((onderwerp_pred == 1) & (onderwerp_true == 1)).sum()
            onderwerp_fp += ((onderwerp_pred == 1) & (onderwerp_true == 0)).sum()
            onderwerp_fn += ((onderwerp_pred == 0) & (onderwerp_true == 1)).sum()

            beleving_tp += ((beleving_pred == 1) & (beleving_true == 1)).sum()
            beleving_fp += ((beleving_pred == 1) & (beleving_true == 0)).sum()
            beleving_fn += ((beleving_pred == 0) & (beleving_true == 1)).sum()

            # Update accuracy metrics
            for j in range(len(onderwerp_names)):
                if onderwerp_pred[j] == onderwerp_true[j]:
                    onderwerp_correct[j] += 1
                onderwerp_total[j] += 1

            for j in range(len(beleving_names)):
                if beleving_pred[j] == beleving_true[j]:
                    beleving_correct[j] += 1
                beleving_total[j] += 1

    # Calculate F1 scores
    epsilon = 1e-8
    onderwerp_precision = onderwerp_tp / (onderwerp_tp + onderwerp_fp + epsilon)
    onderwerp_recall = onderwerp_tp / (onderwerp_tp + onderwerp_fn + epsilon)
    onderwerp_f1_score = 2 * onderwerp_precision * onderwerp_recall / (onderwerp_precision + onderwerp_recall + epsilon)

    beleving_precision = beleving_tp / (beleving_tp + beleving_fp + epsilon)
    beleving_recall = beleving_tp / (beleving_tp + beleving_fn + epsilon)
    beleving_f1_score = 2 * beleving_precision * beleving_recall / (beleving_precision + beleving_recall + epsilon)

    # Calculate accuracies
    onderwerp_acc = onderwerp_correct.sum() / onderwerp_total.sum()
    beleving_acc = beleving_correct.sum() / beleving_total.sum()

    # Get threshold statistics (convert to probability space for human readability)
    if model.use_thresholds:
        onderwerp_thresh_mean = logit_to_prob(model.onderwerp_tau_logit).mean().item()
        onderwerp_thresh_min = logit_to_prob(model.onderwerp_tau_logit).min().item()
        onderwerp_thresh_max = logit_to_prob(model.onderwerp_tau_logit).max().item()
        onderwerp_thresh_std = logit_to_prob(model.onderwerp_tau_logit).std().item()
        beleving_thresh_mean = logit_to_prob(model.beleving_tau_logit).mean().item()
        beleving_thresh_min = logit_to_prob(model.beleving_tau_logit).min().item()
        beleving_thresh_max = logit_to_prob(model.beleving_tau_logit).max().item()
        beleving_thresh_std = logit_to_prob(model.beleving_tau_logit).std().item()
    else:
        # Fixed threshold values
        onderwerp_thresh_mean = onderwerp_thresh_min = onderwerp_thresh_max = onderwerp_thresh_std = 0.5
        beleving_thresh_mean = beleving_thresh_min = beleving_thresh_max = beleving_thresh_std = 0.5

    # Return metrics dictionary
    return {
        'onderwerp_acc': onderwerp_acc,
        'onderwerp_precision': onderwerp_precision,
        'onderwerp_recall': onderwerp_recall,
        'onderwerp_f1': onderwerp_f1_score,
        'beleving_acc': beleving_acc,
        'beleving_precision': beleving_precision,
        'beleving_recall': beleving_recall,
        'beleving_f1': beleving_f1_score,
        'combined_acc': (onderwerp_acc + beleving_acc) / 2,
        'combined_f1': (onderwerp_f1_score + beleving_f1_score) / 2,
        'onderwerp_thresh_mean': onderwerp_thresh_mean,
        'onderwerp_thresh_min': onderwerp_thresh_min,
        'onderwerp_thresh_max': onderwerp_thresh_max,
        'onderwerp_thresh_std': onderwerp_thresh_std,
        'beleving_thresh_mean': beleving_thresh_mean,
        'beleving_thresh_min': beleving_thresh_min,
        'beleving_thresh_max': beleving_thresh_max,
        'beleving_thresh_std': beleving_thresh_std,
        'num_samples_evaluated': num_samples
    }


def grad_l2_norm(params):
    """
    Calculate L2 norm of gradients safely (avoids Python int→Tensor addition).
    Args:
        params: Iterator of parameters (e.g., model.parameters())
    Returns:
        float: L2 norm of all gradients, or 0.0 if no gradients exist
    """
    sq_sum = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        val = g.pow(2).sum()
        sq_sum = val if sq_sum is None else (sq_sum + val)
    if sq_sum is None:
        return 0.0
    return sq_sum.sqrt().item()


def make_opt_sched(model, enc_lr, thr_lr, total_steps, warmup_ratio, eta_min):
    """
    Create optimizer+scheduler for training.
    Optimizer has 1-2 param groups: [0]=encoder+heads, [1]=thresholds (optional).
    """
    # Group 0: encoder + heads
    encoder_params = [p for n, p in model.named_parameters()
                      if not (model.use_thresholds and 'tau_logit' in n)]
    param_groups = [{"params": encoder_params, "lr": enc_lr, "weight_decay": 0.0}]

    # Group 1 (optional): thresholds
    if model.use_thresholds:
        thr_params = [model.onderwerp_tau_logit, model.beleving_tau_logit]
        param_groups.append({"params": thr_params, "lr": thr_lr, "weight_decay": 0.0})

    optimizer = bnb.optim.PagedAdamW8bit(param_groups)
    print("Using bitsandbytes 8-bit AdamW optimizer")

    # Warmup → cosine schedule
    warmup_steps = min(max(1, int(warmup_ratio * total_steps)), max(1, total_steps - 1))
    warmup = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    return optimizer, scheduler


def run_epochs(model, tokenizer, train_loader, val_texts, val_onderwerp, val_beleving,
               onderwerp_names, beleving_names, device,
               *, start_epoch, end_epoch, phase_name="train",
               optimizer, scheduler, temperature, alpha,
               max_length, global_step, amp_dtype=torch.float16):
    """
    Run training for a range of epochs.
    Args:
        model: The model to train
        tokenizer: Tokenizer for text encoding
        train_loader: DataLoader for training batches
        val_texts, val_onderwerp, val_beleving: Validation data
        onderwerp_names, beleving_names: Label names
        device: Device to train on
        start_epoch: Starting epoch (inclusive)
        end_epoch: Ending epoch (exclusive)
        phase_name: Name for logging (default: "train")
        optimizer: Optimizer
        scheduler: LR scheduler
        temperature: Soft-F1 temperature
        alpha: Loss weighting (F1 vs BCE)
        max_length: Max sequence length
        global_step: Starting global step counter
        amp_dtype: Data type for autocasting (default: torch.float16)
    Returns:
        Updated global_step
    """
    num_epochs = end_epoch - start_epoch
    phase_total_steps = max(1, len(train_loader) * num_epochs)

    # Initialize GradScaler for FP16 to prevent underflow
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    model.train()

    for epoch in range(start_epoch, end_epoch):
        total_loss = 0
        total_onderwerp_f1 = 0
        total_beleving_f1 = 0
        total_bce_loss = 0
        total_f1_loss = 0
        num_batches = 0

        print(f"\n[{phase_name.upper()}] Epoch {epoch + 1}/{end_epoch}")
        print("-" * 40)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            onderwerp_labels = batch['onderwerp_labels'].to(device)
            beleving_labels = batch['beleving_labels'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with Autocast (Handles BF16 weights + FP32 Softmax safety)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                onderwerp_logits, beleving_logits = model(input_ids, attention_mask)

                # Calculate Soft-F1 for both heads (conditionally pass thresholds)
                onderwerp_f1 = calculate_soft_f1(
                    onderwerp_logits, onderwerp_labels,
                    model.onderwerp_tau_logit if model.use_thresholds else None,
                    temperature
                )
                beleving_f1 = calculate_soft_f1(
                    beleving_logits, beleving_labels,
                    model.beleving_tau_logit if model.use_thresholds else None,
                    temperature
                )

                # Calculate BCE loss
                # Design choice (POLA):
                # - BCE is computed on raw logits to maintain probability calibration.
                # - Soft-F1 may use a shifted logit (if thresholds ON) to learn F1-friendly boundaries.
                # - If thresholds OFF, Soft-F1 acts directly on logits; there is a single "source of truth".
                # This keeps behavior unsurprising: either (A) calibrated logits + separate boundary learning,
                # or (B) no extra threshold machinery; F1 and BCE both reference the same logits.
                bce_onderwerp = F.binary_cross_entropy_with_logits(onderwerp_logits, onderwerp_labels)
                bce_beleving = F.binary_cross_entropy_with_logits(beleving_logits, beleving_labels)

                # Combined loss
                f1_loss = (1 - onderwerp_f1) + (1 - beleving_f1)
                bce_loss = bce_onderwerp + bce_beleving
                loss = alpha * (f1_loss / 2) + (1 - alpha) * (bce_loss / 2)

            # Periodic logging
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    # Get predictions (convert thresholds from logit-space to prob-space if enabled)
                    onderwerp_probs = torch.sigmoid(onderwerp_logits.float())
                    beleving_probs = torch.sigmoid(beleving_logits.float())
                    if model.use_thresholds:
                        tau_on = logit_to_prob(model.onderwerp_tau_logit)
                        tau_be = logit_to_prob(model.beleving_tau_logit)
                    else:
                        tau_on = torch.full_like(onderwerp_probs[0], 0.5)
                        tau_be = torch.full_like(beleving_probs[0], 0.5)
                    onderwerp_pred = (onderwerp_probs > tau_on).float()
                    beleving_pred = (beleving_probs > tau_be).float()

                    # Log actual optimizer param group LRs
                    lrs = scheduler.get_last_lr()
                    encoder_head_lr = lrs[0]   # Param group 0: encoder + heads
                    threshold_lr = lrs[1] if len(lrs) > 1 else None  # Param group 1: thresholds (optional)

                    # Threshold statistics (convert to probability space for readability)
                    if model.use_thresholds:
                        onderwerp_thresh_mean = logit_to_prob(model.onderwerp_tau_logit).mean().item()
                        onderwerp_thresh_min = logit_to_prob(model.onderwerp_tau_logit).min().item()
                        onderwerp_thresh_max = logit_to_prob(model.onderwerp_tau_logit).max().item()
                        beleving_thresh_mean = logit_to_prob(model.beleving_tau_logit).mean().item()
                        beleving_thresh_min = logit_to_prob(model.beleving_tau_logit).min().item()
                        beleving_thresh_max = logit_to_prob(model.beleving_tau_logit).max().item()
                    else:
                        onderwerp_thresh_mean = onderwerp_thresh_min = onderwerp_thresh_max = 0.5
                        beleving_thresh_mean = beleving_thresh_min = beleving_thresh_max = 0.5

                    print(f"  Batch {batch_idx + 1} | Step {global_step + 1}/{phase_total_steps}:")
                    if threshold_lr is not None:
                        print(f"    Total loss: {loss.item():.4f} (α={alpha} F1 + {1-alpha} BCE) | LR: enc_head={encoder_head_lr:.2e} thresh={threshold_lr:.2e}")
                    else:
                        print(f"    Total loss: {loss.item():.4f} (α={alpha} F1 + {1-alpha} BCE) | LR: enc_head={encoder_head_lr:.2e}")
                    print(f"    F1 loss: {(f1_loss/2).item():.4f} | BCE loss: {(bce_loss/2).item():.4f}")
                    print(f"    Onderwerp F1: {onderwerp_f1.item():.4f} | BCE: {bce_onderwerp.item():.4f} | Thresh: {onderwerp_thresh_mean:.3f} [{onderwerp_thresh_min:.3f}-{onderwerp_thresh_max:.3f}]")
                    print(f"    Beleving F1: {beleving_f1.item():.4f} | BCE: {bce_beleving.item():.4f} | Thresh: {beleving_thresh_mean:.3f} [{beleving_thresh_min:.3f}-{beleving_thresh_max:.3f}]")
                    print(f"    Onderwerp preds: {int(onderwerp_pred.sum())} / {int(onderwerp_labels.sum())} true")
                    print(f"    Beleving preds: {int(beleving_pred.sum())} / {int(beleving_labels.sum())} true")

                    # Log to wandb
                    log_dict = {
                        "phase": phase_name,
                        "train/loss": loss.item(),
                        "train/f1_loss": (f1_loss / 2).item(),
                        "train/bce_loss": (bce_loss / 2).item(),
                        "train/onderwerp_f1": onderwerp_f1.item(),
                        "train/onderwerp_bce": bce_onderwerp.item(),
                        "train/beleving_f1": beleving_f1.item(),
                        "train/beleving_bce": bce_beleving.item(),
                        "train/encoder_head_lr": encoder_head_lr,
                        "train/onderwerp_threshold_mean": onderwerp_thresh_mean,
                        "train/onderwerp_threshold_min": onderwerp_thresh_min,
                        "train/onderwerp_threshold_max": onderwerp_thresh_max,
                        "train/beleving_threshold_mean": beleving_thresh_mean,
                        "train/beleving_threshold_min": beleving_thresh_min,
                        "train/beleving_threshold_max": beleving_thresh_max,
                    }
                    if threshold_lr is not None:
                        log_dict["train/threshold_lr"] = threshold_lr
                    wandb.log(log_dict, step=global_step)

            # Backward pass with Scaler
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
            else:
                loss.backward()

            # Calculate gradient norms
            with torch.no_grad():
                onderwerp_thresh_grad = (model.onderwerp_tau_logit.grad.abs().mean().item()
                                         if model.use_thresholds and model.onderwerp_tau_logit.grad is not None else 0.0)
                beleving_thresh_grad = (model.beleving_tau_logit.grad.abs().mean().item()
                                        if model.use_thresholds and model.beleving_tau_logit.grad is not None else 0.0)

                encoder_grad_norm = grad_l2_norm(model.encoder.parameters())
                onderwerp_head_grad_norm = grad_l2_norm(model.onderwerp_head.parameters())
                beleving_head_grad_norm = grad_l2_norm(model.beleving_head.parameters())
                global_grad_norm = grad_l2_norm(model.parameters())

            # Log gradient norms
            wandb.log({
                "phase": phase_name,
                "grads/threshold_onderwerp": onderwerp_thresh_grad,
                "grads/threshold_beleving": beleving_thresh_grad,
                "grads/encoder": encoder_grad_norm,
                "grads/onderwerp_head": onderwerp_head_grad_norm,
                "grads/beleving_head": beleving_head_grad_norm,
                "grads/global_norm": global_grad_norm,
            }, step=global_step)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights and LR
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            # Update counters
            global_step += 1
            total_loss += loss.item()
            total_onderwerp_f1 += onderwerp_f1.item()
            total_beleving_f1 += beleving_f1.item()
            total_f1_loss += (f1_loss / 2).item()
            total_bce_loss += (bce_loss / 2).item()
            num_batches += 1

        # Epoch summary
        avg_loss = total_loss / max(1, num_batches)
        avg_onderwerp_f1 = total_onderwerp_f1 / max(1, num_batches)
        avg_beleving_f1 = total_beleving_f1 / max(1, num_batches)
        avg_f1_loss = total_f1_loss / max(1, num_batches)
        avg_bce_loss = total_bce_loss / max(1, num_batches)

        # Get current LR for summary
        lrs = scheduler.get_last_lr()
        current_lr = lrs[0]  # Display first group LR

        # Threshold statistics (convert to probability space for readability)
        if model.use_thresholds:
            onderwerp_thresh_mean = logit_to_prob(model.onderwerp_tau_logit).mean().item()
            onderwerp_thresh_std = logit_to_prob(model.onderwerp_tau_logit).std().item()
            beleving_thresh_mean = logit_to_prob(model.beleving_tau_logit).mean().item()
            beleving_thresh_std = logit_to_prob(model.beleving_tau_logit).std().item()
        else:
            onderwerp_thresh_mean = onderwerp_thresh_std = 0.5
            beleving_thresh_mean = beleving_thresh_std = 0.5

        print(f"\n  [{phase_name.upper()}] Epoch {epoch + 1} Summary:")
        print(f"    Average total loss: {avg_loss:.4f} (α={alpha} F1 + {1-alpha} BCE)")
        print(f"    Average F1 loss: {avg_f1_loss:.4f} | Average BCE loss: {avg_bce_loss:.4f}")
        print(f"    Average onderwerp F1: {avg_onderwerp_f1:.4f} | Threshold: {onderwerp_thresh_mean:.3f} (σ={onderwerp_thresh_std:.3f})")
        print(f"    Average beleving F1: {avg_beleving_f1:.4f} | Threshold: {beleving_thresh_mean:.3f} (σ={beleving_thresh_std:.3f})")
        print(f"    Average combined F1: {(avg_onderwerp_f1 + avg_beleving_f1) / 2:.4f}")
        print(f"    Current learning rate: {current_lr:.2e}")

        # Per-epoch validation
        print(f"\n  Running validation on 200 samples...")
        val_metrics = evaluate(
            model, val_texts, val_onderwerp, val_beleving, tokenizer, device,
            onderwerp_names, beleving_names, num_samples=200, max_length=max_length, amp_dtype=amp_dtype
        )

        # Log validation metrics
        wandb.log({
            "phase": phase_name,
            "val/onderwerp_acc": val_metrics['onderwerp_acc'],
            "val/onderwerp_precision": val_metrics['onderwerp_precision'],
            "val/onderwerp_recall": val_metrics['onderwerp_recall'],
            "val/onderwerp_f1": val_metrics['onderwerp_f1'],
            "val/beleving_acc": val_metrics['beleving_acc'],
            "val/beleving_precision": val_metrics['beleving_precision'],
            "val/beleving_recall": val_metrics['beleving_recall'],
            "val/beleving_f1": val_metrics['beleving_f1'],
            "val/combined_acc": val_metrics['combined_acc'],
            "val/combined_f1": val_metrics['combined_f1'],
            "val/onderwerp_threshold_mean": val_metrics['onderwerp_thresh_mean'],
            "val/beleving_threshold_mean": val_metrics['beleving_thresh_mean'],
            "epoch": epoch + 1
        }, step=global_step)

        # Log threshold histograms (convert to probability space for readability)
        if model.use_thresholds:
            wandb.log({
                "phase": phase_name,
                "thresholds/onderwerp": wandb.Histogram(logit_to_prob(model.onderwerp_tau_logit).detach().cpu().numpy()),
                "thresholds/beleving": wandb.Histogram(logit_to_prob(model.beleving_tau_logit).detach().cpu().numpy()),
                "epoch": epoch + 1
            }, step=global_step)

        print(f"    Val onderwerp F1: {val_metrics['onderwerp_f1']:.4f} | Val beleving F1: {val_metrics['beleving_f1']:.4f}")
        print(f"    Val combined F1: {val_metrics['combined_f1']:.4f}")

        # Return to training mode
        model.train()

    return global_step

def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train mmBERT dual-head model on Wim dataset with wandb sweeps.")
    parser.add_argument(
        "--excel_file_path",
        type=str,
        required=True,
        help="Path to the Excel file containing the Wim dataset."
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        required=True,
        help="Wandb API key for authentication."
    )
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.excel_file_path:
        parser.error("--excel_file_path must be specified")
    if not args.wandb_key:
        parser.error("--wandb_key must be specified")
    
    # Enable TensorFloat32 for better performance on modern NVIDIA GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # Initialize device
    device = get_device()

    # ============== CONFIGURATION FOR WANDB SWEEPS ==============
    # Login to wandb
    wandb.login(key=args.wandb_key)

    # Fixed model configuration (not swept)
    model_name = "jhu-clsp/mmBERT-base"

    # Sweepable hyperparameters with defaults
    default_config = dict(
        # Reproducibility
        seed=42,

        # Model architecture
        dropout=0.2,
        initial_threshold=0.565,
        max_length=1408,

        # Training switches
        use_thresholds=False,  # If False: no learnable thresholds; Soft-F1 uses raw logits

        # Training
        encoder_peak_lr=8e-5,
        threshold_lr_mult=5.0,  # Threshold LR = encoder_peak_lr * threshold_lr_mult
        num_epochs=15,
        batch_size=16,

        # Loss function
        alpha=0.15,  # Weight for F1 loss in combined loss (0.5 = balanced)
        temperature=2.0,  # Sigmoid smoothing (lower = softer, higher = sharper)

        # LR schedule
        warmup_ratio=0.1,  # 10% warmup
        min_lr=1e-6,
    )

    # Initialize wandb and get config (allows sweep agent to override defaults)
    wandb.init(project="wim-multilabel-mmbert", config=default_config)
    cfg = wandb.config

    # Set seed for reproducibility (before loading data)
    set_seed(cfg.seed)

    # Load RD dataset
    print("\nLoading dataset...")
    texts, onderwerp, beleving, onderwerp_names, beleving_names = load_wim_dataset(
        excel_path=args.excel_file_path, max_samples=None  # Using full dataset for better training
    )

    print(f"\nDataset loaded:")
    print(f"  Samples: {len(texts)}")
    print(f"  Onderwerp labels: {len(onderwerp_names)}")
    print(f"  Beleving labels: {len(beleving_names)}")
    print(f"  Avg onderwerp per sample: {onderwerp.sum(axis=1).mean():.2f}")
    print(f"  Avg beleving per sample: {beleving.sum(axis=1).mean():.2f}")

    # Unpack hyperparameters from wandb.config
    dropout = cfg.dropout
    initial_threshold = cfg.initial_threshold
    max_length = cfg.max_length
    encoder_peak_lr = cfg.encoder_peak_lr
    threshold_peak_lr = encoder_peak_lr * cfg.threshold_lr_mult  # Derived from multiplier
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    alpha = cfg.alpha
    temperature = cfg.temperature
    warmup_ratio = cfg.warmup_ratio
    min_lr = cfg.min_lr
    # ================================================================

    # Load tokenizer and create model
    print("\nLoading mmBERT-base tokenizer and creating dual-head model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = mmBERTDualHead(
        model_name=model_name,
        num_onderwerp=len(onderwerp_names),
        num_beleving=len(beleving_names),
        dropout=dropout,
        initial_threshold=initial_threshold,
        use_thresholds=cfg.use_thresholds
    )

    # Move model to device
    model = model.to(device)

    # Ensure thresholds match encoder dtype for mixed precision safety
    encoder_dtype = next(model.encoder.parameters()).dtype
    with torch.no_grad():
        if model.use_thresholds:
            model.onderwerp_tau_logit.copy_(model.onderwerp_tau_logit.to(encoder_dtype))
            model.beleving_tau_logit.copy_(model.beleving_tau_logit.to(encoder_dtype))

    print(f"Model loaded and moved to {device} in {encoder_dtype} precision.")
    print(f"  Onderwerp head: {len(onderwerp_names)} outputs")
    print(f"  Beleving head: {len(beleving_names)} outputs")

    amp_dtype = torch.float16  # Using Mixed Precision (FP32 weights + FP16 Ops)
    print(f"Using Mixed Precision Training (Weights: {encoder_dtype}, Ops: {amp_dtype})")
    # Split data into train/val (80/20)
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_onderwerp = onderwerp[:split_idx]
    train_beleving = beleving[:split_idx]
    val_texts = texts[split_idx:]
    val_onderwerp = onderwerp[split_idx:]
    val_beleving = beleving[split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")

    # Create training dataset and dataloader
    train_dataset = DutchDualLabelDataset(
        train_texts, train_onderwerp, train_beleving, tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * num_epochs

    # Log derived/computed values to wandb (sweepable params already in config)
    wandb.config.update({
        # Fixed model configuration
        "model_name": model_name,
        "num_onderwerp": len(onderwerp_names),
        "num_beleving": len(beleving_names),

        # Derived training params
        "threshold_peak_lr": threshold_peak_lr,
        "total_training_steps": total_training_steps,

        # Dataset info
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "total_samples": len(texts),
        "split_ratio": 0.8,

        # Loss configuration (derived from alpha)
        "loss_type": "combined_f1_bce",
        "f1_weight": alpha,
        "bce_weight": 1 - alpha,

        # Fixed features
        "learnable_thresholds": cfg.use_thresholds,
        "per_class_thresholds": cfg.use_thresholds,
        "gradient_clipping": True,
        "max_grad_norm": 1.0,
    }, allow_val_change=True)

    # Print training info
    print(f"\nStarting training for {num_epochs} total epochs with COMBINED F1+BCE LOSS...")
    print(f"Loss formula: {alpha} * (1-F1) + {1-alpha} * BCE")
    print(f"Temperature for Soft-F1: {temperature} | Initial thresholds: {initial_threshold}")
    print(f"Batch size: {batch_size} | Total training batches: {steps_per_epoch}")
    print(f"Learnable thresholds enabled for both onderwerp and beleving heads")
    print("=" * 60)

    # ===== SINGLE-PHASE TRAINING =====
    print(f"\n{'='*60}")
    print(f"TRAINING: {num_epochs} epoch(s)")
    print(f"{'='*60}")

    # Create optimizer and scheduler
    optimizer, scheduler = make_opt_sched(
        model,
        enc_lr=encoder_peak_lr,
        thr_lr=threshold_peak_lr,
        total_steps=total_training_steps,
        warmup_ratio=warmup_ratio,
        eta_min=min_lr
    )

    # Run training
    global_step = run_epochs(
        model, tokenizer, train_loader,
        val_texts, val_onderwerp, val_beleving,
        onderwerp_names, beleving_names, device,
        start_epoch=0, end_epoch=num_epochs,
        phase_name="train",
        optimizer=optimizer, scheduler=scheduler,
        temperature=temperature, alpha=alpha,
        max_length=max_length, global_step=0,
        amp_dtype=amp_dtype
    )

    # Training complete
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    # Final evaluation on larger validation set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)

    print(f"\nEvaluating on 500 validation samples...")
    final_metrics = evaluate(
        model, val_texts, val_onderwerp, val_beleving, tokenizer, device,
        onderwerp_names, beleving_names, num_samples=500, max_length=max_length, amp_dtype=amp_dtype
    )

    # Print overall metrics
    print("\n" + "=" * 60)
    print(f"FINAL METRICS (on {final_metrics['num_samples_evaluated']} validation samples)")
    print("-" * 40)

    print(f"  Onderwerp:")
    print(f"    Accuracy: {final_metrics['onderwerp_acc']:.1%}")
    print(f"    Precision: {final_metrics['onderwerp_precision']:.3f}")
    print(f"    Recall: {final_metrics['onderwerp_recall']:.3f}")
    print(f"    F1 Score: {final_metrics['onderwerp_f1']:.3f}")

    print(f"\n  Beleving:")
    print(f"    Accuracy: {final_metrics['beleving_acc']:.1%}")
    print(f"    Precision: {final_metrics['beleving_precision']:.3f}")
    print(f"    Recall: {final_metrics['beleving_recall']:.3f}")
    print(f"    F1 Score: {final_metrics['beleving_f1']:.3f}")

    print(f"\n  Combined:")
    print(f"    Average Accuracy: {final_metrics['combined_acc']:.1%}")
    print(f"    Average F1: {final_metrics['combined_f1']:.3f}")

    # Log final metrics to wandb
    wandb.log({
        "final/onderwerp_acc": final_metrics['onderwerp_acc'],
        "final/onderwerp_precision": final_metrics['onderwerp_precision'],
        "final/onderwerp_recall": final_metrics['onderwerp_recall'],
        "final/onderwerp_f1": final_metrics['onderwerp_f1'],
        "final/beleving_acc": final_metrics['beleving_acc'],
        "final/beleving_precision": final_metrics['beleving_precision'],
        "final/beleving_recall": final_metrics['beleving_recall'],
        "final/beleving_f1": final_metrics['beleving_f1'],
        "final/combined_acc": final_metrics['combined_acc'],
        "final/combined_f1": final_metrics['combined_f1'],
    }, step=global_step)

    print("\n" + "=" * 60)
    print("Training complete! 🎉")
    print("mmBERT-base dual-head architecture with balanced F1+BCE loss")
    print(f"Loss formula: {alpha} * (1-F1) + {1-alpha} * BCE")
    print(f"Temperature: {temperature}")
    if cfg.use_thresholds:
        print(f"Learned per-class thresholds:")
        print(f"  Onderwerp ({len(onderwerp_names)} classes): mean={final_metrics['onderwerp_thresh_mean']:.3f} [{final_metrics['onderwerp_thresh_min']:.3f}-{final_metrics['onderwerp_thresh_max']:.3f}] σ={final_metrics['onderwerp_thresh_std']:.3f}")
        print(f"  Beleving ({len(beleving_names)} classes): mean={final_metrics['beleving_thresh_mean']:.3f} [{final_metrics['beleving_thresh_min']:.3f}-{final_metrics['beleving_thresh_max']:.3f}] σ={final_metrics['beleving_thresh_std']:.3f}")
    else:
        print("Thresholds disabled (fixed cutoff τ=0.5 for both heads).")
    print(f"With gradient clipping (max_norm=1.0) and warmup LR schedule")
    print(f"Full dataset: {len(texts)} samples | Batch size: {batch_size} | Epochs: {num_epochs}")
    print(f"mmBERT: Modern multilingual encoder (1800+ languages, max_length: {max_length})")

    # Save final model weights (minimal model saving)
    save_path = "./output/mmbert_dual_head_final.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")

    # Save Hugging Face-compatible checkpoint (encoder + tokenizer + custom heads)
    hf_dir = "./output/mmbert_dual_head_hf"
    os.makedirs(hf_dir, exist_ok=True)
    # Save base encoder and tokenizer in HF format
    model.encoder.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    # Save custom heads and metadata alongside
    head_state = {
        "onderwerp_head_state": model.onderwerp_head.state_dict(),
        "beleving_head_state": model.beleving_head.state_dict(),
        "use_thresholds": model.use_thresholds,
        "num_onderwerp": len(onderwerp_names),
        "num_beleving": len(beleving_names),
        "dropout": dropout,
        "max_length": max_length,
        "alpha": alpha,
        "temperature": temperature,
        "model_name": model_name,
    }
    if model.use_thresholds:
        head_state["onderwerp_tau_logit"] = model.onderwerp_tau_logit.detach().cpu()
        head_state["beleving_tau_logit"] = model.beleving_tau_logit.detach().cpu()
    torch.save(head_state, os.path.join(hf_dir, "dual_head_state.pt"))
    # Save label names for convenience
    with open(os.path.join(hf_dir, "label_names.json"), "w") as f:
        json.dump({
            "onderwerp": list(map(str, onderwerp_names)),
            "beleving": list(map(str, beleving_names))
        }, f, ensure_ascii=False, indent=2)
    print(f"HF-compatible checkpoint saved to '{hf_dir}' (encoder+tokenizer), with heads in dual_head_state.pt")

    # Finish wandb run
    wandb.finish()
    print("\nWandB logging completed and run finished.")


if __name__ == "__main__":
    main()