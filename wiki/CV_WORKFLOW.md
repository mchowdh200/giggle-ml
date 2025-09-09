# Cross-Validation and Hyperparameter Optimization Workflow

This document describes the complete workflow for training HyenaDNA with proper cross-validation and hyperparameter optimization.

## Overview

The system implements a rigorous 3-way split of the Roadmap Epigenomics dataset:
- **Training set (70%)**: Used for hyperparameter search and model training
- **Validation set (15%)**: Used for hyperparameter selection during search
- **Test set (15%)**: Used for final unbiased evaluation (use only once!)

The splits are based on tissue+chromatin state pairs (e.g., "Lung_Strong_transcription") to ensure no data leakage between splits.

## Complete Workflow

### 1. Hyperparameter Search
```bash
python scripts/hparam_search.py
```

**What it does:**
- Tests all combinations of learning rates, margins, and batch sizes
- Uses train split for training, validation split for selection
- Saves results to `models/hdna_seqpare_hparam_search/search_results.json`
- Identifies and saves best hyperparameters to `best_hyperparams.json`
- Can be resumed if interrupted (skips already completed combinations)

**Search space (configurable in `src/giggleml/train/hparam_config.py`):**
- Learning rates: [1e-6, 1e-5, 2e-5, 5e-5, 1e-4]
- Margins: [0.5, 1.0, 1.5, 2.0, 3.0] 
- Batch sizes: [10] (currently fixed due to memory constraints)

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

## Manual Usage

You can also run individual components manually:

### Training with specific hyperparameters:
```bash
python src/giggleml/train/hdna_seqpare_ft.py \
  --use_cv \
  --cv_split train \
  --learning_rate 2e-5 \
  --margin 1.0 \
  --clusters_per_batch 10 \
  --validation_freq 5
```

### Evaluation on specific splits:
```bash
# Validate during training
python src/giggleml/train/hdna_seqpare_ft.py --use_cv --cv_split train --validation_freq 2

# Final test evaluation  
python src/giggleml/train/hdna_seqpare_ft.py --use_cv --cv_split test
```

## Key Features

### Data Integrity
- **No data leakage**: Tissue+chromatin state pairs are strictly separated between splits
- **Reproducible splits**: Deterministic hash-based splitting ensures consistent results
- **Validation**: Prevents training on validation/test data

### Robust Hyperparameter Search
- **Grid search**: Systematic evaluation of all combinations
- **Persistence**: Results saved and can be resumed if interrupted
- **Best tracking**: Automatically identifies optimal hyperparameters

### Validation Metrics
- **Triplet Loss**: Standard triplet margin loss
- **Triplet Accuracy**: Percentage of correctly ordered triplets (d(anchor,positive) < d(anchor,negative))

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

## Best Practices

1. **Run hyperparameter search first**: Don't manually tune hyperparameters
2. **Use validation metrics for decisions**: Don't peek at test results during development
3. **Run test evaluation only once**: Treat test set as sacred
4. **Document results**: Save all hyperparameter search results for reproducibility
5. **Monitor for overfitting**: Watch validation vs training metrics during search

## Example Results Format

**Hyperparameter search results:**
```json
{
  "hyperparams": {"learning_rate": 2e-5, "margin": 1.0, "clusters_per_batch": 10},
  "val_loss": 1.234,
  "val_triplet_accuracy": 0.789,
  "train_loss": 1.100,
  "epoch": 10
}
```

**Final test results:**
```json
{
  "hyperparameters": {"learning_rate": 2e-5, "margin": 1.0, "clusters_per_batch": 10},
  "test_loss": 1.250,
  "test_accuracy": 0.776
}
```

This workflow ensures rigorous evaluation while preventing data leakage and test set contamination.