"""
Main training orchestrator for HyenaDNA fine-tuning with graph-based triplets.
Refactored for a clear, type-safe, two-phase initialization lifecycle.
"""

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from lightning_fabric import Fabric
from torch import Tensor, optim
from torch.nn.modules.loss import TripletMarginLoss

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.models.c_model import CModel
from giggleml.train.graph_triplet_miner import GraphTripletMiner
from giggleml.train.rme_clusters_dataset import MiningBatch, RmeSeqpareClusters
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.utils.cv_splits import create_cv_splits
from giggleml.utils.torch_utils import launch_fabric, rprint0
from giggleml.utils.types import GenomicInterval
from giggleml.utils.utils.collection_utils import as_list


@dataclass
class TrainConfig:
    """Consolidated configuration for the training script."""

    mode: str = "train"  # train, val, or test
    total_steps: int = 200
    validation_freq: int = 10
    seed: int = 42
    model: CModel = CModel("16k")
    base_data_dir: Path = Path("data")
    base_model_dir: Path = Path("modelCkpts/cmodel_11072025")
    fasta_path: Path = field(init=False)
    rme_beds_path: Path = field(init=False)
    seqpare_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    margin: float = 3.0
    mining_margin: float | None = None
    mining_strategy: str = "semi-hard"  # hard, semi-hard, or all
    learning_rate: float = 1e-7
    eps: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    batch_size: int = 128
    pk_ratio: float = 2  # p/k, clusters/(positives per cluster)
    density: int = 32  # This is intervals_per_group

    positive_threshold: float = 0.1
    cv_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        }
    )

    clusters_per_batch: int = field(init=False)
    cluster_size: int = field(init=False)
    corrected_batch_size: int = field(init=False)

    dex_batch_size: int = 64
    dex_sub_workers: int = 0

    @staticmethod
    def calculate_graph_args(
        target_batch_size: int, pk_ratio: float
    ) -> tuple[int, int, int]:
        # where batch size is approximately p*k
        cluster_size_f = math.sqrt(target_batch_size / pk_ratio)  # k
        clusters_per_batch_f = cluster_size_f * pk_ratio  # p
        clusters_per_batch = max(1, round(clusters_per_batch_f))
        cluster_size = max(1, round(cluster_size_f))
        corrected_batch_size = clusters_per_batch * cluster_size
        return corrected_batch_size, clusters_per_batch, cluster_size

    def __post_init__(self):
        """Set derived paths and validate parameters after initialization."""
        # --- Parameter Validation ---
        assert self.mode in ["train", "val", "test"], f"Invalid mode: {self.mode}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.pk_ratio > 0, "pk_ratio must be positive"
        assert self.total_steps > 0, "total_steps must be positive"
        assert math.isclose(sum(self.cv_ratios.values()), 1.0), (
            "cv_ratios must sum to 1.0"
        )

        # --- Derived Paths ---
        self.fasta_path = self.base_data_dir / "hg/hg19.fa"
        rme = self.base_data_dir / "roadmap_epigenomics"
        self.rme_beds_path = rme / "beds"
        self.seqpare_dir = rme / "seqpareRanks"
        self.checkpoint_dir = self.base_model_dir / "checkpoints"

        self.corrected_batch_size, self.clusters_per_batch, self.cluster_size = (
            TrainConfig.calculate_graph_args(self.batch_size, self.pk_ratio)
        )


class StepResult(NamedTuple):
    loss: float
    active_triplets: int
    max_triplets: int


class Finetuner:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.fabric: Fabric | None = None
        self.model: CModel | None = None
        self.optimizer: optim.AdamW | None = None

        self.triplet_miner: GraphTripletMiner | None = None
        self.loss_fn: TripletMarginLoss | None = None

        self.train_dataset: RmeSeqpareClusters | None = None
        self.eval_dataset: RmeSeqpareClusters | None = None

        self.start_step = 0
        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return
        self._setup_fabric()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._setup_datasets()
        self._setup_components()
        self._load_checkpoint()
        self._is_setup = True

    @as_list
    def run(self) -> Iterator[tuple[StepResult, StepResult]]:
        self.setup()
        assert self.fabric, "Fabric not initialized"
        assert self.model, "Model not initialized"
        assert self.triplet_miner, "Triplet miner not initialized"
        assert self.loss_fn, "Loss function not initialized"

        assert self.eval_dataset, "Evaluation dataset not initialized"

        if self.config.mode == "test":
            self.fabric.print("Starting evaluation in test-only mode.")
            return self._evaluate()  # Use step=None for a single test run

        assert self.train_dataset, "Training dataset not initialized for train mode"

        self.fabric.print(f"Starting training for {self.config.total_steps} steps.")
        train_data_iter = iter(self.train_dataset)

        for step in range(self.start_step, self.config.total_steps):
            train_result = self._train_step(train_data_iter, step)

            if (step + 1) % self.config.validation_freq == 0:
                eval_result = self._evaluate(step)
                yield train_result, eval_result
                # Save checkpoint at the same frequency as validation
                self._save_checkpoint(step)

        self.fabric.print("Training finished!")

    def _create_dataset(
        self,
        seqpare_db: SeqpareDB,
        allowed_rme_names: list[str],
        target_batch_size: int,
        seed: int,
    ) -> RmeSeqpareClusters:
        assert self.fabric
        _, cluster_amnt, cluster_size = TrainConfig.calculate_graph_args(
            target_batch_size, self.config.pk_ratio
        )
        return RmeSeqpareClusters(
            road_epig_path=self.config.rme_beds_path,
            seqpare=seqpare_db,
            world_size=self.fabric.world_size,
            rank=self.fabric.global_rank,
            positive_threshold=self.config.positive_threshold,
            groups_per_cluster=cluster_size,
            intervals_per_group=self.config.density,
            allowed_rme_names=allowed_rme_names,
            clusters_amnt=cluster_amnt,
            seed=seed,
        )

    def _setup_datasets(self):
        assert self.fabric
        seqpare_db = SeqpareDB(self.config.seqpare_dir)
        cv_splits = create_cv_splits(self.config.rme_beds_path, **self.config.cv_ratios)

        if self.config.mode in ["train", "val"]:
            train_rme_names = cv_splits["train"]
            eval_rme_names = cv_splits["val"]
            self.fabric.print(
                f"Setting up training dataset with {len(train_rme_names)} files."
            )
            self.train_dataset = self._create_dataset(
                seqpare_db=seqpare_db,
                allowed_rme_names=train_rme_names,
                target_batch_size=self.config.batch_size,
                seed=self.config.seed,
            )
        elif self.config.mode == "test":
            eval_rme_names = cv_splits["test"]
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}")

        self.fabric.print(
            f"Setting up evaluation dataset with {len(eval_rme_names)} files."
        )
        # INFO: it's unclear how to adjust subgraph batch_size as bed count changes
        eval_batch_size = self.config.batch_size
        self.eval_dataset = self._create_dataset(
            seqpare_db=seqpare_db,
            allowed_rme_names=eval_rme_names,
            target_batch_size=eval_batch_size,
            seed=self.config.seed + 1,
        )

    def _setup_fabric(self):
        self.fabric = launch_fabric()

    def _setup_components(self):
        assert self.fabric
        model = self.config.model
        optimizer = optim.AdamW(
            params=model.hot_parameters(),
            lr=self.config.learning_rate,
            eps=self.config.eps,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.model.mark_forward_method("distributed_embed")  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
        assert self.model

        self.loss_fn = TripletMarginLoss(margin=self.config.margin, reduction="mean")
        self.triplet_miner = GraphTripletMiner(
            margin=self.config.margin,
            mining_strategy=self.config.mining_strategy,
            mining_margin=self.config.mining_margin,
        )

    def _load_checkpoint(self):
        assert self.fabric and self.model
        checkpoint_path = self.config.checkpoint_dir / "latest-checkpoint.pt"
        if not checkpoint_path.is_file():
            self.fabric.print("No checkpoint found, starting from scratch.")
            return

        self.fabric.print(f"Loading checkpoint: {checkpoint_path}")

        state_to_load: dict[str, Any] = {"model": self.model}

        if self.config.mode in ["train", "val"]:
            assert self.optimizer
            state_to_load["optimizer"] = self.optimizer

        if self.config.mode == "train" and self.train_dataset:
            state_to_load["train_dataset"] = self.train_dataset

        # loads fields dynamically based on dict keys
        remaining_state = self.fabric.load(checkpoint_path, state_to_load)

        if "step" in remaining_state:
            self.start_step = remaining_state["step"] + 1
            self.fabric.print(f"Resuming from step {self.start_step}")

    def _save_checkpoint(self, step: int):
        assert self.fabric and self.model and self.optimizer
        if self.config.mode != "train":
            return

        checkpoint_path = self.config.checkpoint_dir / "latest-checkpoint.pt"

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "step": step,
            "train_dataset": self.train_dataset,
        }

        self.fabric.save(checkpoint_path, state)
        self.fabric.print(f"Checkpoint saved at step {step} to {checkpoint_path}")

    def _run_batch(
        self, data_iter: Iterator[MiningBatch], is_training: bool
    ) -> StepResult:
        """
        Runs a single batch of data through the model.

        Handles both training (backprop) and evaluation (no_grad, all_reduce).
        Returns the calculated loss (float) and number of triplets (int).
        """
        assert self.fabric and self.model and self.triplet_miner and self.loss_fn

        # 1. Set model mode and zero_grad if training
        if is_training:
            assert self.optimizer
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # 2. Data loading: Rank 0 fetches, then broadcasts to all
        if self.fabric.global_rank == 0:
            data_batch = next(data_iter)
        else:
            data_batch = (None, None)  # Placeholder for broadcast

        data_batch = self.fabric.broadcast(data_batch, src=0)
        node_intervals, adj_matrix = data_batch
        assert node_intervals is not None and adj_matrix is not None

        # 3. Call distributed_embed
        batch_tensor = self.model.distributed_embed(
            self._batch_with_fasta(node_intervals),
            self.config.dex_batch_size,
            self.config.dex_sub_workers,
        )

        # 4. Move to device
        batch_tensor = batch_tensor.to(self.fabric.device)
        adj_matrix = adj_matrix.to(self.fabric.device)

        # 5. Mining and Loss Calculation
        with torch.set_grad_enabled(is_training):
            anchor_idx, pos_idx, neg_idx = self.triplet_miner.mine(
                batch_tensor, adj_matrix
            )

            # 2. All-reduce to get the TOTAL triplet count across all ranks
            total_active_triplets = int(
                cast(
                    Tensor,
                    self.fabric.all_reduce(
                        torch.tensor(
                            anchor_idx.numel(),
                            device=self.fabric.device,
                            dtype=torch.long,
                        ),
                        reduce_op="sum",
                    ),
                ).item()
            )

            rprint0("adj, row edges:", adj_matrix.sum(dim=1))
            rprint0("adj, symm diffs?", (adj_matrix.T != adj_matrix).abs().sum())

            if self.config.mining_strategy != "all":
                max_triplets = adj_matrix.shape[0]
            else:
                max_triplets = int(
                    ((adj_matrix.sum(dim=1) - 1) * (adj_matrix == 0).sum(dim=1))
                    .sum()
                    .item()
                )

            # Handle empty batches
            if total_active_triplets == 0:
                self.fabric.print("no triplets found in batch, skipping...")
                return StepResult(0, 0, max_triplets)

            if anchor_idx.numel() > 0:
                anchor_embeds = batch_tensor[anchor_idx]
                positive_embeds = batch_tensor[pos_idx]
                negative_embeds = batch_tensor[neg_idx]
                loss = self.loss_fn(anchor_embeds, positive_embeds, negative_embeds)
            else:
                # This rank has 0 triplets, but others don't.
                # Create a 0.0 loss tensor that's compatible with autograd.

                # This is a hack to ensure autograd traverses the computational graph
                # even in the zero loss case. Necessary, because distributed calls
                # during the backward call (such as with custom autograd functions)
                # must always be called on all ranks.
                loss = batch_tensor.sum() * 0.0

        # 6. Backward pass / Reduction
        if is_training:
            assert self.optimizer
            self.fabric.backward(loss)
            self.optimizer.step()

        # average the loss across all ranks
        avg_loss_tensor = cast(Tensor, self.fabric.all_reduce(loss, reduce_op="mean"))
        final_loss = avg_loss_tensor.item()

        return StepResult(final_loss, total_active_triplets, max_triplets)

    def _train_step(
        self, train_data_iter: Iterator[MiningBatch], step: int
    ) -> StepResult:
        assert self.fabric
        train_result = self._run_batch(train_data_iter, is_training=True)
        train_loss, triplets_found, _ = train_result

        if triplets_found > 0:
            self.fabric.print(
                f"Step {step} | "
                f"Train Loss: {train_loss:.4f} | Active Triplets: {triplets_found}"
            )
            self.fabric.log("train_loss", train_loss, step=step)
            self.fabric.log("triplets_found", triplets_found, step=step)

        return train_result

    def _evaluate(self, step: int | None = None) -> StepResult:
        assert self.fabric
        assert self.eval_dataset
        eval_data_iter = iter(self.eval_dataset)

        train_result = self._run_batch(eval_data_iter, is_training=False)
        loss, active_triplets, _ = train_result

        # --- Logging ---
        eval_split_name = "Test" if self.config.mode == "test" else "Validation"
        log_prefix = "test" if self.config.mode == "test" else "val"

        if step is not None:
            self.fabric.print(
                f"Step {step}"
                f" | {eval_split_name} Loss: {loss:.4f}"
                f" | Active Triplets: {active_triplets}"
            )
            self.fabric.log(f"{log_prefix}_loss", loss, step=step)
        else:
            self.fabric.print(f"{eval_split_name} Loss: {loss:.4f}")

        return train_result

    def _batch_with_fasta(
        self, batch: Iterable[Sequence[GenomicInterval]]
    ) -> list[MemoryIntervalDataset]:
        results = list()

        for item in batch:
            result = MemoryIntervalDataset(item, self.config.fasta_path)
            results.append(result)

        return results
