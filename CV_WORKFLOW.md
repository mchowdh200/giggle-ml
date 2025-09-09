# Cross-Validation and Hyperparameter Optimization Workflow

This document describes the complete workflow for training HyenaDNA with proper cross-validation and comprehensive hyperparameter optimization.

## Overview

The system implements a rigorous 3-way split of the Roadmap Epigenomics dataset:
- **Training set (70%)**: Used for hyperparameter search and model training
- **Validation set (15%)**: Used for hyperparameter selection during search
- **Test set (15%)**: Used for final unbiased evaluation (use only once!)

The splits are based on tissue+chromatin state pairs (e.g., "Lung_Strong_transcription") to ensure no data leakage between splits.

## Complete Workflow

### 1. Hyperparameter Search
```bash
# Full search (30,375 combinations)
python scripts/hparam_search.py

# Conservative search (18 combinations) - recommended for testing
python scripts/hparam_search.py --conservative

# Resume interrupted search
python scripts/hparam_search.py --resume
```

**What it does:**
- Tests all combinations of hyperparameters across 9 dimensions
- Uses train split for training, validation split for selection
- Saves results to `models/hdna_seqpare_hparam_search/search_results.json`
- Identifies and saves best hyperparameters to `best_hyperparams.json`
- Can be resumed if interrupted (skips already completed combinations)
- Supports dataset resumption for reproducibility

### 2. Train Final Model
```bash
python scripts/train_with_best_hparams.py
```

**What it does:**
- Loads best hyperparameters from step 1
- Trains the final model on the full training set using optimal hyperparameters
- Validates during training using validation set
- Saves final model checkpoint

### 3. Test Set Evaluation (FINAL STEP - RUN ONLY ONCE!)
```bash
python scripts/test_evaluation.py
```

**What it does:**
- Prompts for confirmation (to prevent accidental multiple runs)
- Evaluates the final model on the held-out test set
- Provides unbiased performance estimate
- Saves results to `final_test_results.json`

**⚠️ Warning:** This should only be run once after hyperparameter optimization is complete to avoid test set contamination.

## Hyperparameter Search Space

### Default Search Space (30,375 total combinations)
**Core Training Parameters:**
- **Learning rates**: [1e-6, 1e-5, 2e-5, 5e-5, 1e-4] (5 values)
- **Margins**: [0.5, 1.0, 1.5, 2.0, 3.0] (5 values)
- **Epochs**: [8, 10, 12] (3 values)

**Data Sampling Parameters:**
- **Batch sizes**: [8, 10, 12] (clusters per batch) (3 values)
- **Cluster sizes**: [8, 10, 12] (intervals per cluster) (3 values)
- **Densities**: [20, 30, 40] (intervals per candidate) (3 values)

**AdamW Optimizer Parameters:**
- **Beta1**: [0.85, 0.9, 0.95] (3 values)
- **Beta2**: [0.99, 0.999, 0.9999] (3 values)
- **Weight decay**: [0.0, 1e-4, 1e-3] (3 values)

### Conservative Search Space (18 total combinations)
For faster experimentation:
- **Learning rates**: [1e-5, 2e-5, 5e-5] (3 values)
- **Margins**: [1.0, 1.5, 2.0] (3 values)
- **All other parameters**: Fixed to standard values
- **Weight decay**: [0.0, 1e-4] (2 values only)

## Resumption and Robustness Features

### Dataset Resumption
- **Deterministic sampling**: Reproducible random sequences with seeds
- **State tracking**: Can resume from any epoch with identical data
- **Cross-validation integrity**: Maintains proper train/val/test splits

### Search Resumption
- **Automatic detection**: Identifies completed vs partial runs
- **Smart skipping**: Avoids re-running completed combinations
- **Progress preservation**: Saves intermediate results continuously
- **Float precision handling**: Robust hyperparameter comparison

### Data Integrity
- **No data leakage**: Tissue+chromatin state pairs strictly separated
- **Reproducible splits**: Deterministic hash-based splitting
- **Validation**: Prevents training on validation/test data

## Manual Usage

### Training with specific hyperparameters:
```bash
python src/giggleml/train/hdna_seqpare_ft.py \
  --use_cv \
  --cv_split train \
  --learning_rate 2e-5 \
  --margin 1.0 \
  --clusters_per_batch 10 \
  --cluster_size 10 \
  --density 30 \
  --epochs 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --weight_decay 1e-4 \
  --validation_freq 5
```

### Resuming from specific epoch:
```bash
python src/giggleml/train/hdna_seqpare_ft.py \
  --use_cv \
  --cv_split train \
  --resume_from_epoch 5 \
  --learning_rate 2e-5 \
  # ... other hyperparameters
```

## Validation Metrics

- **Triplet Loss**: Standard triplet margin loss
- **Triplet Accuracy**: Percentage of correctly ordered triplets (d(anchor,positive) < d(anchor,negative))
- **Cross-split Generalization**: How well embeddings from train set generalize to val set

## File Structure

```
models/hdna_seqpare_hparam_search/
├── search_results.json           # All hyperparameter search results
├── best_hyperparams.json         # Best hyperparameters identified
└── final_test_results.json       # Final test set evaluation results

scripts/
├── hparam_search.py              # Hyperparameter grid search
├── train_with_best_hparams.py    # Train final model
└── test_evaluation.py            # Final test evaluation

src/giggleml/
├── train/
│   ├── hdna_seqpare_ft.py        # Main training script
│   └── hparam_config.py          # Hyperparameter configuration
└── utils/
    └── cv_splits.py              # Cross-validation split utilities
```

## Customizing Search Space

Edit `src/giggleml/train/hparam_config.py`:

```python
@classmethod
def default(cls) -> "HyperparameterConfig":
    return cls(
        learning_rates=[1e-5, 2e-5, 5e-5],  # Reduce options
        margins=[0.5, 1.0, 2.0],             # Focus on key values
        batch_sizes=[10],                     # Fix if memory limited
        cluster_sizes=[8, 12],                # Experiment with clustering
        densities=[20, 40],                   # Test data density impact
        epochs=[10],                          # Fix for time constraints
        betas_1=[0.9],                        # Standard optimizer
        betas_2=[0.999],                      # Standard optimizer  
        weight_decays=[0.0, 1e-4, 1e-3]      # Test regularization
    )
```

## Best Practices

1. **Start with conservative search**: Use `--conservative` for initial experiments
2. **Use resumption**: Take advantage of `--resume` for long searches
3. **Monitor progress**: Check intermediate results in `search_results.json`
4. **Validate assumptions**: Review best hyperparameters for sanity
5. **Document experiments**: Keep track of search configurations used
6. **Test set discipline**: Only run test evaluation once per experiment

## Performance Considerations

- **Full search time**: ~30,000 combinations × ~10 minutes = ~200 days (distribute across GPUs!)
- **Conservative search**: ~20 combinations × ~10 minutes = ~3 hours
- **Memory usage**: Larger batch sizes may require more GPU memory
- **Resumption overhead**: Minimal, state tracking is lightweight

This comprehensive system ensures rigorous hyperparameter optimization while maintaining scientific rigor and reproducibility.