#!/usr/bin/env python3
"""
Hyperparameter search script for HyenaDNA fine-tuning.
"""

import subprocess
import sys
from pathlib import Path

from giggleml.train.hparam_config import (
    HyperparameterConfig,
    HyperparameterSearchResults,
    ValidationResult
)


def run_training(hyperparams: dict, results_dir: Path, resume_from_epoch: int = None) -> tuple[float, float, float]:
    """
    Run training with given hyperparameters and return validation metrics.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        results_dir: Directory to save results
        resume_from_epoch: Epoch to resume from (for partial runs)
    
    Returns:
        (val_loss, val_accuracy, train_loss)
    """
    # Build command to run training script
    cmd = [
        sys.executable, "src/giggleml/train/hdna_seqpare_ft.py",
        "--use_cv",
        "--cv_split", "train", 
        "--learning_rate", str(hyperparams["learning_rate"]),
        "--margin", str(hyperparams["margin"]),
        "--clusters_per_batch", str(hyperparams["clusters_per_batch"]),
        "--cluster_size", str(hyperparams["cluster_size"]),
        "--density", str(hyperparams["density"]),
        "--epochs", str(hyperparams["epochs"]),
        "--beta1", str(hyperparams["beta1"]),
        "--beta2", str(hyperparams["beta2"]),
        "--weight_decay", str(hyperparams["weight_decay"]),
        "--validation_freq", "2",  # Validate more frequently during search
        "--seed", str(hyperparams.get("seed", 42))
    ]
    
    # Add resumption if specified
    if resume_from_epoch is not None:
        cmd.extend(["--resume_from_epoch", str(resume_from_epoch)])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run training subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with error: {result.stderr}")
        # Return poor metrics to skip this combination
        return 999.0, 0.0, 999.0
    
    # Parse output to extract final validation metrics
    # This is a simple approach - could be improved with structured logging
    lines = result.stdout.split('\n')
    val_loss = None
    val_accuracy = None
    train_loss = None
    
    for line in reversed(lines):
        if "Val Loss:" in line and "Val Acc:" in line:
            try:
                parts = line.split("|")
                val_part = [p for p in parts if "Val Loss:" in p][0]
                acc_part = [p for p in parts if "Val Acc:" in p][0]
                
                val_loss = float(val_part.split("Val Loss:")[1].strip())
                val_accuracy = float(acc_part.split("Val Acc:")[1].strip())
            except (IndexError, ValueError) as e:
                print(f"Could not parse validation metrics: {e}")
                continue
        elif "Train Loss:" in line and train_loss is None:
            try:
                parts = line.split("|")
                train_part = [p for p in parts if "Train Loss:" in p][0]
                train_loss = float(train_part.split("Train Loss:")[1].strip())
            except (IndexError, ValueError):
                continue
        
        if val_loss is not None and val_accuracy is not None and train_loss is not None:
            break
    
    # Default values if parsing failed
    if val_loss is None:
        val_loss = 999.0
    if val_accuracy is None:
        val_accuracy = 0.0
    if train_loss is None:
        train_loss = 999.0
    
    return val_loss, val_accuracy, train_loss


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter search for HyenaDNA")
    parser.add_argument("--conservative", action="store_true", help="Use conservative search space")
    parser.add_argument("--resume", action="store_true", help="Resume partial runs")
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path("models/hdna_seqpare_hparam_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "search_results.json"
    
    # Initialize hyperparameter search
    if args.conservative:
        config = HyperparameterConfig.conservative()
        print("Using conservative hyperparameter search space")
    else:
        config = HyperparameterConfig.default()
        print("Using default hyperparameter search space")
    
    search_results = HyperparameterSearchResults(results_path)
    
    combinations = config.grid_search_combinations()
    print(f"Starting hyperparameter search with {len(combinations)} combinations")
    print(f"Results will be saved to: {results_path}")
    
    # Get already completed combinations to avoid re-running
    print(f"Found {len(search_results.results)} existing results")
    
    # Handle resumption of partial runs
    if args.resume:
        partial_results = search_results.get_partial_results()
        print(f"Found {len(partial_results)} partial results that can be resumed")
    
    # Run grid search
    for i, hyperparams in enumerate(combinations):
        # Add seed to hyperparams for reproducibility
        hyperparams["seed"] = 42
        
        # Check if already completed
        if search_results.is_hyperparams_completed(hyperparams):
            print(f"Skipping combination {i+1}/{len(combinations)} (already completed)")
            continue
        
        print(f"\n=== Combination {i+1}/{len(combinations)} ===")
        print(f"Hyperparameters: {hyperparams}")
        
        # Check for resumption
        resume_from_epoch = None
        if args.resume:
            for result in search_results.results:
                if (result.hyperparams == hyperparams and 
                    result.completed_epoch < result.epoch):
                    resume_from_epoch = result.completed_epoch
                    print(f"Resuming from epoch {resume_from_epoch}")
                    break
        
        try:
            val_loss, val_accuracy, train_loss = run_training(
                hyperparams, results_dir, resume_from_epoch
            )
            
            # Save result
            result = ValidationResult(
                hyperparams=hyperparams,
                val_loss=val_loss,
                val_triplet_accuracy=val_accuracy,
                train_loss=train_loss,
                epoch=hyperparams["epochs"],
                completed_epoch=hyperparams["epochs"]  # Mark as fully completed
            )
            
            search_results.add_result(result)
            print(f"Result: Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}, Train Loss={train_loss:.4f}")
            
        except Exception as e:
            print(f"Error running combination {hyperparams}: {e}")
            continue
    
    # Print best results
    best = search_results.get_best_result()
    if best:
        print(f"\n=== BEST RESULT ===")
        print(f"Hyperparameters: {best.hyperparams}")
        print(f"Validation Loss: {best.val_loss:.4f}")
        print(f"Validation Accuracy: {best.val_triplet_accuracy:.4f}")
        print(f"Train Loss: {best.train_loss:.4f}")
        
        # Save best hyperparameters separately for easy access
        best_path = results_dir / "best_hyperparams.json"
        import json
        with open(best_path, 'w') as f:
            json.dump(best.hyperparams, f, indent=2)
        print(f"Best hyperparameters saved to: {best_path}")
        
        print(f"\n=== NEXT STEPS ===")
        print("1. Train final model with best hyperparameters:")
        print("   python scripts/train_with_best_hparams.py")
        print()
        print("2. Evaluate on test set (only once!):")
        print("   python scripts/test_evaluation.py")
    else:
        print("No successful results found")
    
    print(f"\nHyperparameter search completed. Full results in: {results_path}")


if __name__ == "__main__":
    main()