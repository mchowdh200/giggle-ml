import hashlib
import itertools
from pathlib import Path
from typing import Any

from .roadmap_epigenomics import cell_type_chrm_state_split


def split_rme_names(
    rme_names: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split RME names (tissue+chromatin state pairs) into train/val/test sets.

    Uses deterministic hashing to ensure reproducible splits across runs.

    Args:
        rme_names: List of RME file names (e.g., "Lung_Strong_transcription")
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "train", "val", "test" containing lists of RME names
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    # Create deterministic ordering based on name hash
    def name_hash(name: str) -> int:
        return int(hashlib.md5(f"{name}_{seed}".encode()).hexdigest(), 16)

    sorted_names = sorted(rme_names, key=name_hash)

    n_total = len(sorted_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Handle rounding

    splits = {
        "train": sorted_names[:n_train],
        "val": sorted_names[n_train : n_train + n_val],
        "test": sorted_names[n_train + n_val : n_train + n_val + n_test],
    }

    return splits


def get_available_rme_names(rme_dir: Path) -> list[str]:
    """
    Get list of available RME names by scanning the bed files directory.

    Args:
        rme_dir: Path to roadmap epigenomics beds directory

    Returns:
        List of RME names (without .bed extension)
    """
    rme_names = []

    bed_files = itertools.chain(rme_dir.glob("*.bed"), rme_dir.glob("*.bed.gz"))

    for bed_file in bed_files:
        # Remove .bed extension
        name = bed_file.stem
        # Validate it's a proper tissue+chromatin state combination
        try:
            _ = cell_type_chrm_state_split(name)
            rme_names.append(name)
        except ValueError:
            # Skip files that don't match expected naming pattern
            continue

    return sorted(rme_names)


def create_cv_splits(rme_dir: Path, **kwargs: Any) -> dict[str, list[str]]:
    """
    Convenience function to create CV splits from RME directory.

    Args:
        rme_dir: Path to roadmap epigenomics beds directory
        **kwargs: Additional arguments passed to split_rme_names

    Returns:
        Dict with train/val/test splits
    """
    available_names = get_available_rme_names(rme_dir)
    return split_rme_names(available_names, **kwargs)

