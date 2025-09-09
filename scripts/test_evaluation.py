#!/usr/bin/env python3
"""
Final test set evaluation using the best hyperparameters.
This should only be run once after hyperparameter optimization is complete.
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
    
    print("=== FINAL TEST SET EVALUATION ===")
    print("Using best hyperparameters from validation:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")
    
    print("\nWARNING: This evaluation should only be run once after hyperparameter")
    print("optimization is complete to avoid test set contamination.")
    
    response = input("\nProceed with test evaluation? (y/N): ")
    if response.lower() != 'y':
        print("Test evaluation cancelled.")
        sys.exit(0)
    
    # Run evaluation on test set
    cmd = [
        sys.executable, "src/giggleml/train/hdna_seqpare_ft.py",
        "--use_cv",
        "--cv_split", "test",
        "--learning_rate", str(best_hparams["learning_rate"]),
        "--margin", str(best_hparams["margin"]),
        "--clusters_per_batch", str(best_hparams["clusters_per_batch"]),
        "--validation_freq", "1"  # Validate every epoch for test evaluation
    ]
    
    print(f"\nRunning test evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Test evaluation failed!")
        print("Error:", result.stderr)
        sys.exit(1)
    
    # Parse and save test results
    lines = result.stdout.split('\n')
    test_loss = None
    test_accuracy = None
    
    for line in reversed(lines):
        if "Val Loss:" in line and "Val Acc:" in line:  # "Val" metrics on test set
            try:
                parts = line.split("|")
                val_part = [p for p in parts if "Val Loss:" in p][0]
                acc_part = [p for p in parts if "Val Acc:" in p][0]
                
                test_loss = float(val_part.split("Val Loss:")[1].strip())
                test_accuracy = float(acc_part.split("Val Acc:")[1].strip())
                break
            except (IndexError, ValueError):
                continue
    
    # Save test results
    test_results = {
        "hyperparameters": best_hparams,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    
    test_results_path = results_dir / "final_test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n=== FINAL TEST RESULTS ===")
    if test_loss is not None and test_accuracy is not None:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nResults saved to: {test_results_path}")
    else:
        print("Could not parse test metrics from output")
        print("Full output:")
        print(result.stdout)
    
    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()