#!/usr/bin/env python3
"""
Hyperparameter search script for HyenaDNA fine-tuning.
"""

from pathlib import Path
from typing import Any

from giggleml.train.hparam_config import (
    HyperparameterConfig,
    HyperparameterSearchResults,
    ValidationResult,
)
from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils.torch_utils import get_rank


def print0(*args: Any, **kwargs: Any) -> None:
    if get_rank() != 0:
        return
    print(*args, **kwargs)


def run_training_with_hyperparams(
    hyperparams: dict[str, Any], results_dir: Path
) -> float:
    """
    Run training on validation set with given hyperparameters for hyperparameter optimization.

    Args:
        hyperparams: Dictionary of hyperparameters
        results_dir: Directory to save results

    Returns:
        val_loss
    """
    try:
        print0(f"Running hyperparameter optimization with hyperparams: {hyperparams}")

        # Create TrainConfig with hyperparameters
        config = TrainConfig(
            mode="val",
            learning_rate=hyperparams["learning_rate"],
            margin=hyperparams["margin"],
            batch_size=hyperparams["batch_size"],
            pk_ratio=hyperparams["pk_ratio"],
            density=hyperparams["density"],
            epochs=hyperparams["epochs"],
            beta1=hyperparams["beta1"],
            beta2=hyperparams["beta2"],
            weight_decay=hyperparams["weight_decay"],
            validation_freq=1,  # Validate every epoch during search
            seed=hyperparams.get("seed", 42),
            base_model_dir=results_dir,
        )

        # Create and run finetuner
        finetuner = Finetuner(config)
        val_loss = finetuner.run()
        return val_loss

    except Exception as e:
        print(f"Hyperparameter optimization failed with error: {e}")
        # Re-raise the exception to fail explicitly
        raise


def main():
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter search for HyenaDNA")
    parser.add_argument(
        "--conservative", action="store_true", help="Use conservative search space"
    )
    parser.add_argument("--resume", action="store_true", help="Resume partial runs")
    args = parser.parse_args()

    # Setup paths
    results_dir = Path("models/hdna_seqpare_hparam_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "search_results.json"

    # Initialize hyperparameter search
    if args.conservative:
        config = HyperparameterConfig.conservative()
        print0("Using conservative hyperparameter search space")
    else:
        config = HyperparameterConfig.default()
        print0("Using default hyperparameter search space")

    search_results = HyperparameterSearchResults(results_path)

    combinations = config.grid_search_combinations()
    print0(f"Starting hyperparameter search with {len(combinations)} combinations")
    print0(f"Results will be saved to: {results_path}")

    # Get already completed combinations to avoid re-running
    print0(f"Found {len(search_results.results)} existing results")

    # Handle resumption of partial runs
    if args.resume:
        partial_results = search_results.get_partial_results()
        print0(f"Found {len(partial_results)} partial results that can be resumed")

    # Run grid search
    for i, hyperparams in enumerate(combinations):
        # Add seed to hyperparams for reproducibility
        hyperparams["seed"] = 42

        # Check if already completed
        if search_results.is_hyperparams_completed(hyperparams):
            print0(
                f"Skipping combination {i + 1}/{len(combinations)} (already completed)"
            )
            continue

        print0(f"\n=== Combination {i + 1}/{len(combinations)} ===")
        print0(f"Hyperparameters: {hyperparams}")

        # Note: Resumption is now handled automatically by the new API through checkpoint loading
        if args.resume:
            # Check if there are partial results for logging purposes
            for result in search_results.results:
                if (
                    result.hyperparams == hyperparams
                    and result.completed_epoch < result.epoch
                ):
                    print0("Found partial result that may be resumed automatically")
                    break

        try:
            val_loss = run_training_with_hyperparams(hyperparams, results_dir)

            # Save result (train_loss set to 0.0 since we're only evaluating)
            result = ValidationResult(
                hyperparams=hyperparams,
                val_loss=val_loss,
                train_loss=0.0,  # Not applicable for evaluation-only mode
                epoch=hyperparams["epochs"],
                completed_epoch=hyperparams["epochs"],  # Mark as fully completed
            )

            search_results.add_result(result)
            print0(f"Result: Val Loss={val_loss:.4f}")

        except Exception as e:
            print(f"Error running combination {hyperparams}: {e}")
            raise

    if get_rank() == 0:
        # Print best results
        best = search_results.get_best_result()
        if best:
            print0("\n=== BEST RESULT ===")
            print0(f"Hyperparameters: {best.hyperparams}")
            print0(f"Validation Loss: {best.val_loss:.4f}")

            # Save best hyperparameters separately for easy access
            best_path = results_dir / "best_hyperparams.json"
            import json

            with open(best_path, "w") as f:
                json.dump(best.hyperparams, f, indent=2)
            print0(f"Best hyperparameters saved to: {best_path}")
        else:
            print0("No successful results found")

    print0(f"\nHyperparameter search completed. Full results in: {results_path}")


if __name__ == "__main__":
    main()
