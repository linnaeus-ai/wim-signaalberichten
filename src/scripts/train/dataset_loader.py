"""
Dataset loader for multi-label classification.
Carmack-style: minimal abstraction, direct data flow, fast operations.
"""

import numpy as np
from datasets import load_dataset


def load_wim_dataset(excel_path, test_size=0.2, max_samples=None):
    """
    Load dataset and encode multi-labels.
    
    Dataset contains Dutch municipal complaint conversations with two types of labels:
    - onderwerp: What the message is about
    - beleving: How the citizen experienced the interaction
    Args:
        excel_path: Path to the Excel file containing the dataset
        test_size: Fraction of data to reserve for testing (default: 0.2)
        max_samples: Limit number of samples (None = all samples)
    Returns:
        texts: List of conversation strings
        onderwerp_encoded: numpy array [n_samples, n_onderwerp] - multi-hot encoded topics
        beleving_encoded: numpy array [n_samples, n_beleving] - multi-hot encoded experiences
        onderwerp_labels: List of onderwerp label names (sorted alphabetically)
        beleving_labels: List of beleving label names (sorted alphabetically)
    """

    # Load RD dataset
    print(f"Loading dataset from Excel file: {excel_path}")
    ds = load_dataset('csv', data_files=excel_path, split='train')
    
    # Keep only essential columns from RD dataset
    ds= ds.select_columns(['text', 'onderwerp_labels', 'beleving_labels'])
    print(f"Loaded: {len(ds)} samples from Excel")
    
    # Shuffle with fixed seed for reproducibility
    ds = ds.shuffle(seed=42)
    print(f"Shuffled combined dataset")

    # Replace "No subtopic found" with empty list (for both onderwerp and beleving)
    ds = ds.map(lambda x: {
        **x,
        'onderwerp_labels': [] if x['onderwerp_labels'] == ['No subtopic found'] else x['onderwerp_labels'],
        'beleving_labels': [] if x['beleving_labels'] == ['No subtopic found'] else x['beleving_labels']
    })
    no_onderwerp_count = sum(1 for sample in ds if len(sample['onderwerp_labels']) == 0)
    no_beleving_count = sum(1 for sample in ds if len(sample['beleving_labels']) == 0)
    print(f"Replaced 'No subtopic found' with empty list: {no_onderwerp_count} onderwerp, {no_beleving_count} beleving")

    # Limit samples if requested
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Limited to {len(ds)} samples")
    
    # Split into train and test
    split_dataset = ds.train_test_split(test_size=test_size, seed=42)
    ds_train = split_dataset['train']
    ds_test = split_dataset['test']

    print(f"Split dataset: {len(ds_train)} train samples, {len(ds_test)} test (test_size={test_size})")

    # Extract all unique labels from the entire dataset
    onderwerp_set = set()
    beleving_set = set()

    for sample in ds:
        for label in sample['onderwerp_labels'].split(';'):
            onderwerp_set.add(label)
        for label in sample['beleving_labels'].split(';'):
            beleving_set.add(label)

    # Sort labels alphabetically for consistent indexing across runs
    onderwerp_labels = sorted(onderwerp_set)
    beleving_labels = sorted(beleving_set)

    print(f"Found {len(onderwerp_labels)} unique onderwerp labels")
    print(f"Found {len(beleving_labels)} unique beleving labels")

    # Create label -> index mappings
    onderwerp_to_idx = {label: idx for idx, label in enumerate(onderwerp_labels)}
    beleving_to_idx = {label: idx for idx, label in enumerate(beleving_labels)}

    # Encode labels to multi-hot vectors
    n_samples = len(ds)
    n_onderwerp = len(onderwerp_labels)
    n_beleving = len(beleving_labels)

    # Preallocate arrays (faster than appending)
    texts = []
    onderwerp_encoded = np.zeros((n_samples, n_onderwerp), dtype=np.float32)
    beleving_encoded = np.zeros((n_samples, n_beleving), dtype=np.float32)

    # Fill arrays
    for i, sample in enumerate(ds):
        texts.append(sample['text'])

        # Encode onderwerp labels (multi-hot)
        for label in sample['onderwerp_labels'].split(';'):
            idx = onderwerp_to_idx[label]
            onderwerp_encoded[i, idx] = 1.0

        # Encode beleving labels (multi-hot)
        for label in sample['beleving_labels'].split(';'):
            idx = beleving_to_idx[label]
            beleving_encoded[i, idx] = 1.0

    print(f"Encoded {n_samples} samples")
    print(f"  onderwerp shape: {onderwerp_encoded.shape}")
    print(f"  beleving shape: {beleving_encoded.shape}")

    return texts, onderwerp_encoded, beleving_encoded, onderwerp_labels, beleving_labels


def print_sample_info(texts, onderwerp_encoded, beleving_encoded,
                     onderwerp_labels, beleving_labels, sample_idx=0):
    """
    Print information about a specific sample (useful for debugging).
    Args:
        All outputs from load_wim_dataset()
        sample_idx: Which sample to print (default: 0)
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE {sample_idx}")
    print(f"{'='*60}")
    print(f"Text: {texts[sample_idx][:200]}...")
    print()

    # Get active onderwerp labels
    onderwerp_active = [onderwerp_labels[i] for i, val in enumerate(onderwerp_encoded[sample_idx]) if val == 1]
    print(f"Onderwerp labels ({len(onderwerp_active)}):")
    for label in onderwerp_active:
        print(f"  - {label}")
    print()

    # Get active beleving labels
    beleving_active = [beleving_labels[i] for i, val in enumerate(beleving_encoded[sample_idx]) if val == 1]
    print(f"Beleving labels ({len(beleving_active)}):")
    for label in beleving_active:
        print(f"  - {label}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the loader
    print("Testing UWV dataset loader...\n")

    # Load small subset for testing
    texts, onderwerp, beleving, onderwerp_names, beleving_names = load_wim_dataset(max_samples=10)

    # Print first sample
    print_sample_info(texts, onderwerp, beleving, onderwerp_names, beleving_names, sample_idx=0)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(texts)}")
    print(f"  Avg onderwerp labels per sample: {onderwerp.sum(axis=1).mean():.2f}")
    print(f"  Avg beleving labels per sample: {beleving.sum(axis=1).mean():.2f}")
    print(f"  Text length range: {min(len(t) for t in texts)} - {max(len(t) for t in texts)} chars")
