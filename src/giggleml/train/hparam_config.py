import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search space."""

    learning_rates: list[float]
    margins: list[float]
    batch_sizes: list[int] | None = None  # clusters_per_batch
    cluster_sizes: list[int] | None = None  # intervals per cluster
    densities: list[int] | None = None  # intervals per candidate (centroid_size)
    positive_threshold: float | None = None
    epochs: list[int] | None = None  # training epochs
    # AdamW parameters
    betas_1: list[float] | None = None  # AdamW beta1
    betas_2: list[float] | None = None  # AdamW beta2
    weight_decays: list[float] | None = None  # AdamW weight decay

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [10]
        if self.cluster_sizes is None:
            self.cluster_sizes = [10]
        if self.densities is None:
            self.densities = [30]
        if self.positive_threshold is None:
            self.positive_threshold = 0.7
        if self.epochs is None:
            self.epochs = [10]
        if self.betas_1 is None:
            self.betas_1 = [0.9]
        if self.betas_2 is None:
            self.betas_2 = [0.999]
        if self.weight_decays is None:
            self.weight_decays = [0.0]

    @classmethod
    def default(cls) -> "HyperparameterConfig":
        """Default hyperparameter search space."""
        return cls(
            learning_rates=[1e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            margins=[0.5, 1.0, 1.5, 2.0, 3.0],
            batch_sizes=[8, 10, 12],  # Expand if memory allows
            cluster_sizes=[8, 10, 12],  # Intervals per cluster
            densities=[20, 30, 40],  # Intervals per candidate
            positive_threshold=0.7,  # Fixed
            epochs=[8, 10, 12],  # Training epochs
            # AdamW hyperparameters
            betas_1=[0.85, 0.9, 0.95],  # AdamW beta1
            betas_2=[0.99, 0.999, 0.9999],  # AdamW beta2
            weight_decays=[0.0, 1e-4, 1e-3],  # Weight decay
        )

    @classmethod
    def conservative(cls) -> "HyperparameterConfig":
        """Conservative search space for faster experimentation."""
        return cls(
            learning_rates=[1e-5, 2e-5, 5e-5],
            margins=[1.0, 1.5, 2.0],
            batch_sizes=[10],  # Keep fixed for memory
            cluster_sizes=[10],  # Keep standard
            densities=[30],  # Keep standard
            positive_threshold=0.7,  # Fixed
            epochs=[10],  # Keep standard
            betas_1=[0.9],  # Standard
            betas_2=[0.999],  # Standard
            weight_decays=[0.0, 1e-4],  # Light regularization only
        )

    def grid_search_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations for grid search."""
        combinations = []

        # Ensure all lists are not None after __post_init__
        assert self.batch_sizes is not None
        assert self.cluster_sizes is not None
        assert self.densities is not None
        assert self.epochs is not None
        assert self.betas_1 is not None
        assert self.betas_2 is not None
        assert self.weight_decays is not None

        for (
            lr,
            margin,
            batch_size,
            cluster_size,
            density,
            epochs,
            beta1,
            beta2,
            weight_decay,
        ) in itertools.product(
            self.learning_rates,
            self.margins,
            self.batch_sizes,
            self.cluster_sizes,
            self.densities,
            self.epochs,
            self.betas_1,
            self.betas_2,
            self.weight_decays,
        ):
            combinations.append(
                {
                    "learning_rate": lr,
                    "margin": margin,
                    "clusters_per_batch": batch_size,
                    "cluster_size": cluster_size,
                    "density": density,
                    "epochs": epochs,
                    "beta1": beta1,
                    "beta2": beta2,
                    "weight_decay": weight_decay,
                }
            )
        return combinations


@dataclass
class ValidationResult:
    """Results from hyperparameter validation."""

    hyperparams: dict[str, Any]
    val_loss: float
    val_triplet_accuracy: float
    train_loss: float
    epoch: int
    completed_epoch: int = 0  # For resumption tracking
    dataset_state: dict[str, Any] | None = None  # Dataset state for resumption

    def __post_init__(self):
        if self.dataset_state is None:
            self.dataset_state = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperparams": self.hyperparams,
            "val_loss": self.val_loss,
            "val_triplet_accuracy": self.val_triplet_accuracy,
            "train_loss": self.train_loss,
            "epoch": self.epoch,
            "completed_epoch": self.completed_epoch,
            "dataset_state": self.dataset_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        # Handle backward compatibility
        if "completed_epoch" not in data:
            data["completed_epoch"] = data.get("epoch", 0)
        if "dataset_state" not in data:
            data["dataset_state"] = {}
        return cls(**data)


class HyperparameterSearchResults:
    """Manages saving/loading hyperparameter search results."""

    def __init__(self, results_path: Path):
        self.results_path: Path = results_path
        self.results: list[ValidationResult] = []
        self.load_existing()

    def load_existing(self):
        """Load existing results if file exists."""
        if self.results_path.exists():
            try:
                with open(self.results_path, "r") as f:
                    data = json.load(f)
                self.results = [ValidationResult.from_dict(r) for r in data]
                print(f"Loaded {len(self.results)} existing hyperparameter results")
            except Exception as e:
                print(f"Could not load existing results: {e}")
                self.results = []

    def add_result(self, result: ValidationResult):
        """Add new result and save to disk."""
        self.results.append(result)
        self.save()

    def save(self):
        """Save results to disk."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def get_completed_hyperparams(self) -> set:
        """Get set of hyperparameter combinations already completed."""
        completed = set()
        for result in self.results:
            # Convert dict to frozenset for hashing, handling float precision
            hp_items = []
            for key, value in sorted(result.hyperparams.items()):
                if isinstance(value, float):
                    # Round floats for consistent hashing
                    hp_items.append((key, round(value, 10)))
                else:
                    hp_items.append((key, value))
            completed.add(tuple(hp_items))
        return completed

    def get_partial_results(self) -> list[ValidationResult]:
        """Get results that may be partially completed (for resumption)."""
        return [r for r in self.results if r.completed_epoch < r.epoch]

    def is_hyperparams_completed(self, hyperparams: dict[str, Any]) -> bool:
        """Check if specific hyperparameter combination is fully completed."""
        hp_items = []
        for key, value in sorted(hyperparams.items()):
            if isinstance(value, float):
                hp_items.append((key, round(value, 10)))
            else:
                hp_items.append((key, value))
        hp_tuple = tuple(hp_items)

        for result in self.results:
            result_items = []
            for key, value in sorted(result.hyperparams.items()):
                if isinstance(value, float):
                    result_items.append((key, round(value, 10)))
                else:
                    result_items.append((key, value))
            result_tuple = tuple(result_items)

            if hp_tuple == result_tuple and result.completed_epoch >= result.epoch:
                return True
        return False

    def get_best_result(self) -> ValidationResult | None:
        """Get best result based on validation loss."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.val_loss)

    def get_best_hyperparams(self) -> dict[str, Any] | None:
        """Get hyperparameters of best result."""
        best = self.get_best_result()
        return best.hyperparams if best else None
