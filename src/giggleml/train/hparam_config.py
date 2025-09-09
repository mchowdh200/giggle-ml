from dataclasses import dataclass
from typing import Any, Dict, List
import json
import itertools
from pathlib import Path


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search space."""
    learning_rates: List[float]
    margins: List[float] 
    batch_sizes: List[int] = None  # clusters_per_batch
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [10]  # Default from current config
    
    @classmethod
    def default(cls) -> "HyperparameterConfig":
        """Default hyperparameter search space."""
        return cls(
            learning_rates=[1e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            margins=[0.5, 1.0, 1.5, 2.0, 3.0],
            batch_sizes=[10]  # Keep fixed for now due to memory constraints
        )
    
    def grid_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        combinations = []
        for lr, margin, batch_size in itertools.product(
            self.learning_rates, self.margins, self.batch_sizes
        ):
            combinations.append({
                "learning_rate": lr,
                "margin": margin,
                "clusters_per_batch": batch_size
            })
        return combinations


@dataclass 
class ValidationResult:
    """Results from hyperparameter validation."""
    hyperparams: Dict[str, Any]
    val_loss: float
    val_triplet_accuracy: float
    train_loss: float
    epoch: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hyperparams": self.hyperparams,
            "val_loss": self.val_loss,
            "val_triplet_accuracy": self.val_triplet_accuracy,
            "train_loss": self.train_loss,
            "epoch": self.epoch
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        return cls(**data)


class HyperparameterSearchResults:
    """Manages saving/loading hyperparameter search results."""
    
    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.results: List[ValidationResult] = []
        self.load_existing()
    
    def load_existing(self):
        """Load existing results if file exists."""
        if self.results_path.exists():
            try:
                with open(self.results_path, 'r') as f:
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
        with open(self.results_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
    
    def get_completed_hyperparams(self) -> set:
        """Get set of hyperparameter combinations already completed."""
        completed = set()
        for result in self.results:
            # Convert dict to frozenset for hashing
            hp_items = tuple(sorted(result.hyperparams.items()))
            completed.add(hp_items)
        return completed
    
    def get_best_result(self) -> ValidationResult | None:
        """Get best result based on validation loss."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.val_loss)
    
    def get_best_hyperparams(self) -> Dict[str, Any] | None:
        """Get hyperparameters of best result."""
        best = self.get_best_result()
        return best.hyperparams if best else None