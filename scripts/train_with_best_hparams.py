#!/usr/bin/env python3
"""
Train the model using the best hyperparameters found from hyperparameter search.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    # Load best hyperparameters
    results_dir = Path("models/hdna_seqpare_hparam_search")
    best_path = results_dir / "best_hyperparams.json"
    
    if not best_path.exists():
        print("Best hyperparameters not found. Run hyperparameter search first:")
        print("  python scripts/hparam_search.py")
        sys.exit(1)
    
    with open(best_path, 'r') as f:
        best_hparams = json.load(f)
    
    print("Using best hyperparameters:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")
    
    # Run training with best hyperparameters on full training set
    cmd = [
        sys.executable, "src/giggleml/train/hdna_seqpare_ft.py",
        "--use_cv",
        "--cv_split", "train",
        "--learning_rate", str(best_hparams["learning_rate"]),
        "--margin", str(best_hparams["margin"]),
        "--clusters_per_batch", str(best_hparams["clusters_per_batch"]),
        "--cluster_size", str(best_hparams["cluster_size"]),
        "--density", str(best_hparams["density"]),
        "--epochs", str(best_hparams["epochs"]),
        "--beta1", str(best_hparams["beta1"]),
        "--beta2", str(best_hparams["beta2"]),
        "--weight_decay", str(best_hparams["weight_decay"])
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nTraining completed successfully!")
        print("\n=== NEXT STEP ===")
        print("To evaluate on test set (final step, run only once!):")
        print("  python scripts/test_evaluation.py")
        print()
        print("This will use the test split to provide an unbiased performance estimate.")
    else:
        print("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()