#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-shot & one-shot multi-label classificatie met BART MNLI

Functionaliteit:
- Data inladen vanuit CSV.
- Shuffle + optional sampling.
- Train/test split (2/3 – 1/3).
- Zero-shot scores m.b.v. facebook/bart-large-mnli.
- Zero-shot: per-label threshold tuning op train-set via F1-grid search.
- One-shot: per label threshold op basis van één positieve train-voorbeelddocument.
- Evaluatie op test-set (macro/micro/weighted F1, per-label metrics).
- Confusion matrices per label.
- Distributie van labels in train vs test als PNG.
- Export van alle resultaten naar een Excel met 2 sheets:
    1) Per-label metrics (zero-shot én one-shot).
    2) Globale summary metrics (zero-shot én one-shot).

LET OP:
- Pas de standaard pad/kolomnamen aan via command line argumenten
  of door de defaults hieronder te wijzigen.
"""

import argparse
import ast
import time
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm.auto import tqdm


# -----------------------------
# Helper-functies
# -----------------------------

def normalize_label_cell(cell: Any) -> List[str]:
    """
    Zorgt ervoor dat een cel met labels altijd als een lijst wordt teruggegeven.

    Mogelijke inputs:
    - list  -> blijft list
    - NaN   -> lege list
    - scalar (string / andere) -> lijst met 1 element
    """
    def _clean_items(items: List[Any]) -> List[str]:
        cleaned = []
        for item in items:
            if pd.isna(item):
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, tuple):
        cell = list(cell)
    if isinstance(cell, list):
        return _clean_items(cell)
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        stripped = cell.strip()
        if not stripped:
            return []
        parsed = None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, list):
            return _clean_items(parsed)
        if isinstance(parsed, str):
            parsed = parsed.strip()
            return [parsed] if parsed else []
        for sep in (";", ",", "|"):
            if sep in stripped:
                parts = [p.strip() for p in stripped.split(sep)]
                return [p for p in parts if p]
        return [stripped]
    text = str(cell).strip()
    return [text] if text else []


def collect_all_labels(df: pd.DataFrame,
                       onderwerp_col: str,
                       beleving_col: str) -> (List[str], List[str], List[str]):
    """
    Haal alle unieke labels uit onderwerp- en beleving-kolommen.

    Returns:
        all_onderwerp_labels: gesorteerde lijst met alle onderwerp-labels
        all_beleving_labels: gesorteerde lijst met alle beleving-labels
        all_labels: gecombineerde gesorteerde lijst met unieke labels
    """
    all_onderwerp = set()
    all_beleving = set()

    for _, row in df[[onderwerp_col, beleving_col]].iterrows():
        onderwerp_list = normalize_label_cell(row[onderwerp_col])
        beleving_list = normalize_label_cell(row[beleving_col])

        all_onderwerp.update(onderwerp_list)
        all_beleving.update(beleving_list)

    all_onderwerp_labels = sorted(list(all_onderwerp))
    all_beleving_labels = sorted(list(all_beleving))
    all_labels = sorted(list(all_onderwerp.union(all_beleving)))

    return all_onderwerp_labels, all_beleving_labels, all_labels


def get_true_vectors(df: pd.DataFrame,
                     all_labels: List[str],
                     onderwerp_col: str,
                     beleving_col: str) -> np.ndarray:
    """
    Maak een binaire matrix [n_samples, n_labels] waarbij 1 betekent dat het label
    aanwezig is in ofwel onderwerp_labels of beleving_labels voor die rij.
    """
    label_index = {lbl: i for i, lbl in enumerate(all_labels)}
    true_vectors = []

    for _, row in df[[onderwerp_col, beleving_col]].iterrows():
        vec = [0] * len(all_labels)

        onderwerp_list = normalize_label_cell(row[onderwerp_col])
        beleving_list = normalize_label_cell(row[beleving_col])

        for lbl in onderwerp_list + beleving_list:
            if lbl in label_index:
                vec[label_index[lbl]] = 1

        true_vectors.append(vec)

    return np.array(true_vectors, dtype=int)


def get_scores_texts(classifier_pipeline,
                     texts: pd.Series,
                     labels: List[str]) -> List[Dict[str, float]]:
    """
    Haal zero-shot scores op voor een reeks teksten.

    Returns:
        List van dicts: per tekst {label: score}
    """
    text_list = list(texts)

    # Generator functie voor input
    def text_generator():
        for t in text_list:
            yield t

    # Hugging Face pipeline ondersteunt batch processing.
    # Door een generator te gebruiken en te itereren, kunnen we tqdm gebruiken.
    # batch_size=8 is een veilige startwaarde; pas aan indien OOM op GPU.
    results = classifier_pipeline(text_generator(), labels, multi_label=True, batch_size=8)
    scores = []

    # Wrap de iterator in tqdm voor een progress bar
    for result in tqdm(results, total=len(text_list), desc="Scoring"):
        label_scores = {lbl: sc for lbl, sc in zip(result["labels"], result["scores"])}
        scores.append(label_scores)

    return scores


def get_score_matrix(scores: List[Dict[str, float]],
                     all_labels: List[str]) -> np.ndarray:
    """
    Zet een list van label-score dicts om in een matrix [n_samples, n_labels].
    """
    score_matrix = np.zeros((len(scores), len(all_labels)), dtype=float)
    label_index = {lbl: i for i, lbl in enumerate(all_labels)}

    for i, label_score in enumerate(scores):
        for lbl, sc in label_score.items():
            j = label_index.get(lbl)
            if j is not None:
                score_matrix[i, j] = sc

    return score_matrix


def tune_thresholds_zero_shot(train_score_matrix: np.ndarray,
                              train_true: np.ndarray,
                              all_labels: List[str],
                              n_steps: int = 201) -> List[float]:
    """
    Grid search over thresholds per label op de train-set.
    Thresholds lopen van 0 t/m 1 met n_steps.

    Returns:
        List met beste thresholds per label.
    """
    thresholds_grid = np.linspace(0.0, 1.0, n_steps)
    n_labels = train_score_matrix.shape[1]
    best_thresholds = []

    for j in range(n_labels):
        f1s = []
        y_true = train_true[:, j]
        scores = train_score_matrix[:, j]

        for t in thresholds_grid:
            preds = (scores >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            f1s.append(f1)

        best_idx = int(np.argmax(f1s))
        best_thresholds.append(float(thresholds_grid[best_idx]))

    return best_thresholds


def tune_thresholds_one_shot(train_score_matrix: np.ndarray,
                             train_true: np.ndarray,
                             zero_shot_thresholds: List[float]) -> List[float]:
    """
    One-shot thresholds: gebruik per label de score van het
    eerste positieve train-voorbeeld als threshold.

    Als er geen positieve voorbeelden zijn voor een label:
    fallback naar de zero-shot threshold (grid search resultaat).
    """
    n_labels = train_score_matrix.shape[1]
    one_shot_thresholds = []

    for j in range(n_labels):
        positives = np.where(train_true[:, j] == 1)[0]
        if len(positives) == 0:
            # Geen positief voorbeeld -> gebruik zero-shot threshold
            one_shot_thresholds.append(zero_shot_thresholds[j])
        else:
            idx0 = positives[0]
            one_shot_thresholds.append(float(train_score_matrix[idx0, j]))

    return one_shot_thresholds


def apply_thresholds(score_matrix: np.ndarray,
                     thresholds: List[float]) -> np.ndarray:
    """
    Pas per-label thresholds toe op een score matrix.

    Returns:
        Binaire predictiematrix [n_samples, n_labels].
    """
    thresholds_arr = np.array(thresholds)[np.newaxis, :]
    preds = (score_matrix >= thresholds_arr).astype(int)
    return preds


def compute_confusion_matrices(true_matrix: np.ndarray,
                               pred_matrix: np.ndarray,
                               all_labels: List[str]) -> Dict[str, np.ndarray]:
    """
    Maak een confusion matrix per label.
    """
    conf_matrices = {}
    for i, lbl in enumerate(all_labels):
        cm = confusion_matrix(true_matrix[:, i], pred_matrix[:, i])
        conf_matrices[lbl] = cm
    return conf_matrices


def label_distribution_plot(train_true: np.ndarray,
                            test_true: np.ndarray,
                            all_labels: List[str],
                            output_path: str) -> None:
    """
    Plot en sla een PNG op van de label-distributie (train vs test).
    """
    train_counts = train_true.sum(axis=0)
    test_counts = test_true.sum(axis=0)

    x = np.arange(len(all_labels))
    width = 0.4

    plt.figure(figsize=(max(8, len(all_labels) * 0.4), 6))
    plt.bar(x - width / 2, train_counts, width, label="Train")
    plt.bar(x + width / 2, test_counts, width, label="Test")
    plt.xticks(x, all_labels, rotation=45, ha="right")
    plt.ylabel("Aantal positieve voorbeelden")
    plt.title("Label-distributie in Train vs Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_to_excel(output_path: str,
                    all_labels: List[str],
                    zs_thresholds: List[float],
                    os_thresholds: List[float],
                    report_zs: Dict[str, Any],
                    report_os: Dict[str, Any],
                    conf_zs: Dict[str, np.ndarray],
                    conf_os: Dict[str, np.ndarray],
                    summary_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Exporteer alle resultaten naar een Excel bestand met 2 sheets:
    - 'per_label': per-label metrics (zero-shot én one-shot).
    - 'summary': globale metrics (zero-shot én one-shot).
    """

    # Per-label sheet
    data = {
        "label": [],
        "zs_threshold": [],
        "zs_f1": [],
        "zs_precision": [],
        "zs_recall": [],
        "zs_support": [],
        "zs_tp": [],
        "zs_tn": [],
        "zs_fp": [],
        "zs_fn": [],
        "os_threshold": [],
        "os_f1": [],
        "os_precision": [],
        "os_recall": [],
        "os_support": [],
        "os_tp": [],
        "os_tn": [],
        "os_fp": [],
        "os_fn": [],
    }

    for i, lbl in enumerate(all_labels):
        # Zero-shot metrics per label
        zs_metrics = report_zs.get(lbl, {})
        zs_f1 = zs_metrics.get("f1-score", None)
        zs_prec = zs_metrics.get("precision", None)
        zs_rec = zs_metrics.get("recall", None)
        zs_support = zs_metrics.get("support", None)
        cm_zs = conf_zs.get(lbl)
        if cm_zs is not None and cm_zs.shape == (2, 2):
            zs_tn, zs_fp, zs_fn, zs_tp = cm_zs.ravel()
        else:
            zs_tp = zs_tn = zs_fp = zs_fn = 0

        # One-shot metrics per label
        os_metrics = report_os.get(lbl, {})
        os_f1 = os_metrics.get("f1-score", None)
        os_prec = os_metrics.get("precision", None)
        os_rec = os_metrics.get("recall", None)
        os_support = os_metrics.get("support", None)
        cm_os = conf_os.get(lbl)
        if cm_os is not None and cm_os.shape == (2, 2):
            os_tn, os_fp, os_fn, os_tp = cm_os.ravel()
        else:
            os_tp = os_tn = os_fp = os_fn = 0

        data["label"].append(lbl)
        data["zs_threshold"].append(zs_thresholds[i])
        data["zs_f1"].append(zs_f1)
        data["zs_precision"].append(zs_prec)
        data["zs_recall"].append(zs_rec)
        data["zs_support"].append(zs_support)
        data["zs_tp"].append(zs_tp)
        data["zs_tn"].append(zs_tn)
        data["zs_fp"].append(zs_fp)
        data["zs_fn"].append(zs_fn)
        data["os_threshold"].append(os_thresholds[i])
        data["os_f1"].append(os_f1)
        data["os_precision"].append(os_prec)
        data["os_recall"].append(os_rec)
        data["os_support"].append(os_support)
        data["os_tp"].append(os_tp)
        data["os_tn"].append(os_tn)
        data["os_fp"].append(os_fp)
        data["os_fn"].append(os_fn)

    df_per_label = pd.DataFrame(data)

    # Summary sheet
    # summary_metrics heeft vorm:
    # {
    #   "zero_shot": {"macro_f1": ..., "micro_f1": ..., ...},
    #   "one_shot": {...}
    # }
    rows = []
    for model_name, metrics in summary_metrics.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df_summary = pd.DataFrame(rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_per_label.to_excel(writer, sheet_name="per_label", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)


# -----------------------------
# Main pipeline
# -----------------------------

def run_experiment(
    data_path: str = None,
    text_col: str = "text",
    onderwerp_col: str = "onderwerp_labels",
    beleving_col: str = "beleving_labels",
    sample_size: Optional[int] = None,
    train_frac: float = 2.0 / 3.0,
    model_name: str = "facebook/bart-large-mnli",
    random_state: int = 42,
    output_prefix: str = "results",
    hf_dataset_name: str = "UWV/wim-synthetic-data-rd",
    hf_split: str = "train",
) -> None:
    """
    Draai de volledige zero-shot & one-shot pipeline.

    Args:
        data_path: Pad naar input CSV met minimaal de kolommen text, onderwerp_labels, beleving_labels.
            Laat None om automatisch een HuggingFace dataset te gebruiken.
        text_col: Kolomnaam met de tekst.
        onderwerp_col: Kolomnaam met onderwerp-labels (list of scalar).
        beleving_col: Kolomnaam met beleving-labels (list of scalar).
        sample_size: Aantal rijen om te gebruiken (na shufflen). Laat None voor volledige dataset.
        train_frac: Fractie voor train-deel (2/3 default).
        model_name: HuggingFace modelnaam voor zero-shot classificatie.
        random_state: Seed voor reproducible shuffling/splits.
        output_prefix: Prefix voor outputbestanden (Excel, PNG).
        hf_dataset_name: Naam van de HuggingFace dataset voor input.
        hf_split: Welke split van de HuggingFace dataset wordt gebruikt.
    """
    start_time = time.time()

    # Data inladen
    needed_cols = [text_col, onderwerp_col, beleving_col]
    if data_path:
        print(f"Data inladen vanaf CSV: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"HuggingFace dataset laden: {hf_dataset_name} (split: {hf_split})")
        hf_dataset = load_dataset(hf_dataset_name, split=hf_split)
        available_cols = set(hf_dataset.column_names)
        missing_in_hf = [col for col in needed_cols if col not in available_cols]
        if missing_in_hf:
            raise ValueError(
                f"Kolommen {missing_in_hf} niet gevonden in HuggingFace dataset {hf_dataset_name} ({hf_split})."
            )
        hf_dataset = hf_dataset.select_columns(needed_cols)
        df = hf_dataset.to_pandas()
    missing_cols = [col for col in needed_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolommen {missing_cols} niet gevonden in dataset.")

    for col in (onderwerp_col, beleving_col):
        df[col] = df[col].apply(normalize_label_cell)

    initial_rows = len(df)
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    if len(df) == 0:
        raise ValueError("Na opschonen van lege teksten blijven er geen rijen over.")
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Let op: {removed_rows} rijen verwijderd vanwege lege/NaN teksten.")

    
# Shuffle
# Shuffle
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Sample
    if sample_size is not None and sample_size > 0 and sample_size < len(df_shuffled):
        sample = df_shuffled.head(sample_size).copy()
    else:
        sample = df_shuffled.copy()

    n = len(sample)
    train_size = int(train_frac * n)
    if train_size <= 0 or train_size >= n:
        raise ValueError("Train/test split is ongeldig. Check train_frac en sample_size.")

    train_df = sample.iloc[:train_size].copy()
    test_df = sample.iloc[train_size:].copy()

    # Alle labels verzamelen
    all_onderwerp_labels, all_beleving_labels, all_labels = collect_all_labels(
        sample, onderwerp_col, beleving_col
    )

    if len(all_labels) == 0:
        raise ValueError("Geen labels gevonden in onderwerp/beleving kolommen.")

    print(f"Aantal samples totaal: {n}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Aantal unieke labels: {len(all_labels)}")

    # Zero-shot pipeline laden
    print(f"Model laden: {model_name}")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    # Scores ophalen
    print("Zero-shot scores berekenen (train)...")
    train_scores = get_scores_texts(classifier, train_df[text_col], all_labels)
    print("Zero-shot scores berekenen (test)...")
    test_scores = get_scores_texts(classifier, test_df[text_col], all_labels)

    # True vectors
    train_true = get_true_vectors(train_df, all_labels, onderwerp_col, beleving_col)
    test_true = get_true_vectors(test_df, all_labels, onderwerp_col, beleving_col)

    # Score matrices
    train_score_matrix = get_score_matrix(train_scores, all_labels)
    test_score_matrix = get_score_matrix(test_scores, all_labels)

    # Zero-shot thresholds (grid search)
    print("Thresholds tunen (zero-shot, grid search)...")
    zs_thresholds = tune_thresholds_zero_shot(train_score_matrix, train_true, all_labels)

    # One-shot thresholds
    print("Thresholds bepalen (one-shot)...")
    os_thresholds = tune_thresholds_one_shot(train_score_matrix, train_true, zs_thresholds)

    # Predictions zero-shot & one-shot
    zs_pred = apply_thresholds(test_score_matrix, zs_thresholds)
    os_pred = apply_thresholds(test_score_matrix, os_thresholds)

    # Indexen voor onderwerp/beleving subsets
    all_label_index = {lbl: i for i, lbl in enumerate(all_labels)}
    onderwerp_idx = [all_label_index[lbl] for lbl in all_onderwerp_labels]
    beleving_idx = [all_label_index[lbl] for lbl in all_beleving_labels]

    # Metrics zero-shot
    zs_macro_f1 = f1_score(test_true, zs_pred, average="macro", zero_division=0)
    zs_micro_f1 = f1_score(test_true, zs_pred, average="micro", zero_division=0)
    zs_weighted_f1 = f1_score(test_true, zs_pred, average="weighted", zero_division=0)
    zs_macro_f1_onderwerp = f1_score(
        test_true[:, onderwerp_idx], zs_pred[:, onderwerp_idx],
        average="macro", zero_division=0
    )
    zs_macro_f1_beleving = f1_score(
        test_true[:, beleving_idx], zs_pred[:, beleving_idx],
        average="macro", zero_division=0
    )
    report_zs = classification_report(
        test_true, zs_pred, target_names=all_labels,
        zero_division=0, output_dict=True
    )
    conf_zs = compute_confusion_matrices(test_true, zs_pred, all_labels)

    # Metrics one-shot
    os_macro_f1 = f1_score(test_true, os_pred, average="macro", zero_division=0)
    os_micro_f1 = f1_score(test_true, os_pred, average="micro", zero_division=0)
    os_weighted_f1 = f1_score(test_true, os_pred, average="weighted", zero_division=0)
    os_macro_f1_onderwerp = f1_score(
        test_true[:, onderwerp_idx], os_pred[:, onderwerp_idx],
        average="macro", zero_division=0
    )
    os_macro_f1_beleving = f1_score(
        test_true[:, beleving_idx], os_pred[:, beleving_idx],
        average="macro", zero_division=0
    )
    report_os = classification_report(
        test_true, os_pred, target_names=all_labels,
        zero_division=0, output_dict=True
    )
    conf_os = compute_confusion_matrices(test_true, os_pred, all_labels)

    # Summary metrics dict
    summary_metrics = {
        "zero_shot": {
            "macro_f1": zs_macro_f1,
            "micro_f1": zs_micro_f1,
            "weighted_f1": zs_weighted_f1,
            "macro_f1_onderwerp": zs_macro_f1_onderwerp,
            "macro_f1_beleving": zs_macro_f1_beleving,
        },
        "one_shot": {
            "macro_f1": os_macro_f1,
            "micro_f1": os_micro_f1,
            "weighted_f1": os_weighted_f1,
            "macro_f1_onderwerp": os_macro_f1_onderwerp,
            "macro_f1_beleving": os_macro_f1_beleving,
        },
    }

    # Print kernmetrics naar console
    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.2f} seconds")

    print("\nZERO-SHOT METRICS:")
    print(f"Macro-F1: {zs_macro_f1:.4f}")
    print(f"Micro-F1: {zs_micro_f1:.4f}")
    print(f"Weighted-F1: {zs_weighted_f1:.4f}")
    print(f"Macro-F1 onderwerp_labels: {zs_macro_f1_onderwerp:.4f}")
    print(f"Macro-F1 beleving_labels: {zs_macro_f1_beleving:.4f}")

    print("\nONE-SHOT METRICS:")
    print(f"Macro-F1: {os_macro_f1:.4f}")
    print(f"Micro-F1: {os_micro_f1:.4f}")
    print(f"Weighted-F1: {os_weighted_f1:.4f}")
    print(f"Macro-F1 onderwerp_labels: {os_macro_f1_onderwerp:.4f}")
    print(f"Macro-F1 beleving_labels: {os_macro_f1_beleving:.4f}")

    # Label distributie plot
    png_path = f"{output_prefix}_label_distribution.png"
    print(f"\nLabel distributie plot opslaan naar: {png_path}")
    label_distribution_plot(train_true, test_true, all_labels, png_path)

    # Excel export
    excel_path = f"{output_prefix}_metrics.xlsx"
    print(f"Excel resultaten opslaan naar: {excel_path}")
    export_to_excel(
        excel_path,
        all_labels,
        zs_thresholds,
        os_thresholds,
        report_zs,
        report_os,
        conf_zs,
        conf_os,
        summary_metrics,
    )

    print("\nKlaar. Alle resultaten zijn geëxporteerd.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot & one-shot multi-label classificatie met BART MNLI."
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Pad naar input CSV bestand. Laat leeg om HuggingFace dataset te gebruiken."
    )
    parser.add_argument(
        "--text_col", type=str, default="text",
        help="Kolomnaam met tekst (default: 'text')."
    )
    parser.add_argument(
        "--onderwerp_col", type=str, default="onderwerp_labels",
        help="Kolomnaam met onderwerp-labels (default: 'onderwerp_labels')."
    )
    parser.add_argument(
        "--beleving_col", type=str, default="beleving_labels",
        help="Kolomnaam met beleving-labels (default: 'beleving_labels')."
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Aantal te gebruiken rijen na shufflen (laat leeg voor volledige dataset)."
    )
    parser.add_argument(
        "--train_frac", type=float, default=2.0 / 3.0,
        help="Train-fractie (default: 2/3)."
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/bart-large-mnli",
        help="HuggingFace modelnaam (default: facebook/bart-large-mnli)."
    )
    parser.add_argument(
        "--hf_dataset_name", type=str, default="UWV/wim-synthetic-data-rd",
        help="Naam van de HuggingFace dataset voor input (default: 'UWV/wim-synthetic-data-rd')."
    )
    parser.add_argument(
        "--hf_split", type=str, default="train",
        help="Welke split van de HuggingFace dataset (default: 'train')."
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--output_prefix", type=str, default="results",
        help="Prefix voor output bestanden (default: 'results')."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_path=args.data_path,
        text_col=args.text_col,
        onderwerp_col=args.onderwerp_col,
        beleving_col=args.beleving_col,
        sample_size=args.sample_size,
        train_frac=args.train_frac,
        model_name=args.model_name,
        random_state=args.random_state,
        output_prefix=args.output_prefix,
        hf_dataset_name=args.hf_dataset_name,
        hf_split=args.hf_split,
    )
