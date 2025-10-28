"""
Main training orchestrator for HyenaDNA fine-tuning with graph-based triplets.
Refactored for a clear, type-safe, two-phase initialization lifecycle.
"""

import math
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from lightning_fabric import Fabric
from lightning_fabric.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, optim
from torch.nn.modules.loss import TripletMarginLoss

from giggleml.data_wrangling.interval_dataset import MemoryIntervalDataset
from giggleml.models.m_model import MModel
from giggleml.train.graph_triplet_miner import GraphTripletMiner
from giggleml.train.rme_clusters_dataset import MiningBatch, RmeSeqpareClusters
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.utils.cv_splits import create_cv_splits
from giggleml.utils.types import GenomicInterval


@dataclass
class TrainConfig:
    """Consolidated configuration for the training script."""

    mode: str = "train"
    epochs: int = 10
    validation_freq: int = 5
    seed: int = 42
    base_data_dir: Path = Path("data")
    base_model_dir: Path = Path("modelCkpts/mmodel_10272025")
    fasta_path: Path = field(init=False)
    rme_beds_path: Path = field(init=False)
    seqpare_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    margin: float = 3.0
    learning_rate: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0

    batch_size: int = 100
    pk_ratio: float = 1.0
    density: int = 30  # This is intervals_per_group

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

    dex_batch_size: int = 64
    dex_sub_workers: int = 4

    def __post_init__(self):
        """Set derived paths and validate parameters after initialization."""
        # --- Parameter Validation ---
        assert self.mode in ["train", "val", "test"], f"Invalid mode: {self.mode}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.pk_ratio > 0, "pk_ratio must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert math.isclose(sum(self.cv_ratios.values()), 1.0), (
            "cv_ratios must sum to 1.0"
        )

        # --- Derived Paths ---
        self.fasta_path = self.base_data_dir / "hg/hg19.fa"
        self.rme_beds_path = self.base_data_dir / "roadmap_epigenomics/beds"
        self.seqpare_dir = self.base_data_dir / "roadmap_epigenomics/seqpareRanks"
        self.log_dir = self.base_model_dir / "logs"
        self.checkpoint_dir = self.base_model_dir / "checkpoints"

        # where batch size is approximately p*k
        cluster_size_f = math.sqrt(self.batch_size / self.pk_ratio)  # k
        clusters_per_batch_f = cluster_size_f * self.pk_ratio  # p
        self.clusters_per_batch = max(1, round(clusters_per_batch_f))
        self.cluster_size = max(1, round(cluster_size_f))


class Finetuner:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.fabric: Fabric | None = None
        self.model: MModel | None = None
        self.optimizer: optim.AdamW | None = None

        self.triplet_miner: GraphTripletMiner | None = None
        self.loss_fn: TripletMarginLoss | None = None

        self.train_dataset: RmeSeqpareClusters | None = None
        self.eval_dataset: RmeSeqpareClusters | None = None

        self.start_epoch = 0
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

    def run(self) -> float:
        self.setup()
        assert self.fabric, "Fabric not initialized"
        assert self.model, "Model not initialized"
        assert self.triplet_miner, "Triplet miner not initialized"
        assert self.loss_fn, "Loss function not initialized"

        assert self.eval_dataset, "Evaluation dataset not initialized"

        if self.config.mode == "test":
            self.fabric.print("Starting evaluation in test-only mode.")
            return self._evaluate()

        assert self.train_dataset, "Training dataset not initialized for train mode"

        self.fabric.print(f"Starting training for {self.config.epochs} epochs.")
        train_data_iter = iter(self.train_dataset)
        final_eval_loss: float = -1.0

        for epoch in range(self.start_epoch, self.config.epochs):
            self._train_step(train_data_iter, epoch)

            if (epoch + 1) % self.config.validation_freq == 0:
                final_eval_loss = self._evaluate(epoch)

            self._save_checkpoint(epoch)

        self.fabric.print("Training finished!")
        return final_eval_loss

    def _create_dataset(
        self,
        seqpare_db: SeqpareDB,
        allowed_rme_names: list[str],
        clusters_amnt: int,
        seed: int,
    ) -> RmeSeqpareClusters:
        assert self.fabric
        return RmeSeqpareClusters(
            road_epig_path=self.config.rme_beds_path,
            seqpare=seqpare_db,
            world_size=self.fabric.world_size,
            rank=self.fabric.global_rank,
            positive_threshold=self.config.positive_threshold,
            groups_per_cluster=self.config.cluster_size,
            intervals_per_group=self.config.density,
            allowed_rme_names=allowed_rme_names,
            clusters_amnt=clusters_amnt,
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
                clusters_amnt=self.config.clusters_per_batch,
                seed=self.config.seed,
            )
        elif self.config.mode == "test":
            eval_rme_names = cv_splits["test"]
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}")

        self.fabric.print(
            f"Setting up evaluation dataset with {len(eval_rme_names)} files."
        )
        eval_clusters_amnt = (
            2 * self.fabric.world_size
            if self.config.mode == "val"
            else max(5, self.config.clusters_per_batch // 2)
        )
        self.eval_dataset = self._create_dataset(
            seqpare_db=seqpare_db,
            allowed_rme_names=eval_rme_names,
            clusters_amnt=eval_clusters_amnt,
            seed=self.config.seed + 1,
        )

    def _setup_fabric(self):
        logger = TensorBoardLogger(root_dir=self.config.log_dir)
        world_size = int(os.environ.get("WORLD_SIZE") or 1)
        self.fabric = Fabric(
            accelerator="auto", strategy="auto", devices=world_size, loggers=[logger]
        )
        self.fabric.launch()

    def _setup_components(self):
        assert self.fabric
        model = MModel("16k")
        optimizer = optim.AdamW(
            params=model.hot_parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.model.mark_forward_method("distributed_embed")  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
        assert self.model

        self.loss_fn = TripletMarginLoss(margin=self.config.margin, reduction="mean")
        self.triplet_miner = GraphTripletMiner(
            margin=self.config.margin, distance_metric="euclidean"
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

        if "epoch" in remaining_state:
            self.start_epoch = remaining_state["epoch"] + 1
            self.fabric.print(f"Resuming from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch: int):
        assert self.fabric and self.model and self.optimizer
        if self.config.mode != "train":
            return

        checkpoint_path = self.config.checkpoint_dir / "latest-checkpoint.pt"

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": epoch,
            "train_dataset": self.train_dataset,
        }

        self.fabric.save(checkpoint_path, state)
        self.fabric.print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def _train_step(self, train_data_iter: Iterator[MiningBatch], step: int):
        assert self.fabric and self.model and self.optimizer
        assert self.triplet_miner and self.loss_fn

        self.model.train()
        self.optimizer.zero_grad()

        # --- Data loading: Rank 0 fetches, then broadcasts to all ---
        if self.fabric.global_rank == 0:
            data_batch = next(train_data_iter)
        else:
            data_batch = (None, None)  # Placeholder for broadcast

        # Broadcast the (node_intervals, adj_matrix) tuple from rank 0
        data_batch = self.fabric.broadcast(data_batch, src=0)
        node_intervals, adj_matrix = data_batch
        assert node_intervals is not None
        assert adj_matrix is not None
        # All ranks now have the identical node_intervals and adj_matrix

        # Call distributed_embed on all ranks with the *full* interval list.
        # Assumes distributed_embed handles internal sharding and returns
        # the identical, complete tensor on all ranks.
        batch_tensor = self.model.distributed_embed(
            self._batch_with_fasta(node_intervals),
            self.config.dex_batch_size,
            self.config.dex_sub_workers,
        )

        # Move to the correct device
        batch_tensor = batch_tensor.to(self.fabric.device)
        adj_matrix = adj_matrix.to(self.fabric.device)

        # Sanity check
        assert adj_matrix.shape[0] == len(batch_tensor), (
            f"Adjacency matrix shape ({adj_matrix.shape[0]}) does not match "
            f"embeddings shape ({batch_tensor.shape[0]})"
        )

        # --- In-line Mining and Loss Calculation (on flat tensor) ---
        # All ranks do this simultaneously on identical data
        anchor_idx, pos_idx, neg_idx = self.triplet_miner.mine(batch_tensor, adj_matrix)
        assert len(anchor_idx) != 0

        anchor_embeds = batch_tensor[anchor_idx]
        positive_embeds = batch_tensor[pos_idx]
        negative_embeds = batch_tensor[neg_idx]
        loss = self.loss_fn(anchor_embeds, positive_embeds, negative_embeds)

        self.fabric.backward(loss)
        self.optimizer.step()

        train_loss = loss.item()
        triplets_found = anchor_idx.numel()
        self.fabric.print(
            f"Batch {step} | Rank {self.fabric.global_rank} | "
            f"Train Loss: {train_loss:.4f} | Triplets: {triplets_found}"
        )
        self.fabric.log("train_loss", train_loss, step=step)

    def _evaluate(self, step: int | None = None) -> float:
        assert self.fabric and self.model and self.triplet_miner and self.loss_fn
        assert self.eval_dataset

        self.model.eval()
        eval_data_iter = iter(self.eval_dataset)

        # --- Data loading: Rank 0 fetches, then broadcasts to all ---
        if self.fabric.global_rank == 0:
            data_batch = next(eval_data_iter)
        else:
            data_batch = (None, None)  # Placeholder for broadcast

        # Broadcast the (node_intervals, adj_matrix) tuple from rank 0
        data_batch = self.fabric.broadcast(data_batch, src=0)
        node_intervals, adj_matrix = data_batch
        assert node_intervals is not None
        assert adj_matrix is not None
        # All ranks now have the identical node_intervals and eval_adj_matrix

        # Call distributed_embed on all ranks with the *full* interval list.
        batch_tensor = self.model.distributed_embed(
            self._batch_with_fasta(node_intervals),
            self.config.dex_batch_size,
            self.config.dex_sub_workers,
        )

        # Move to the correct device
        batch_tensor = batch_tensor.to(self.fabric.device)
        adj_matrix = adj_matrix.to(self.fabric.device)

        with torch.no_grad():
            anchor_idx, pos_idx, neg_idx = self.triplet_miner.mine(
                batch_tensor, adj_matrix
            )
            assert len(anchor_idx) != 0

            anchor_embeds = batch_tensor[anchor_idx]
            positive_embeds = batch_tensor[pos_idx]
            negative_embeds = batch_tensor[neg_idx]
            loss_tensor = self.loss_fn(anchor_embeds, positive_embeds, negative_embeds)

        # This reduction is still necessary to get a single, averaged loss
        # value from across all ranks (which calculated the same loss,
        # but reduction is good practice and correct).
        avg_loss_tensor: Tensor = cast(
            Tensor, self.fabric.all_reduce(loss_tensor, reduce_op="mean")
        )
        loss = avg_loss_tensor.item()

        # --- Logging ---
        eval_split_name = "Test" if self.config.mode == "test" else "Validation"
        log_prefix = "test" if self.config.mode == "test" else "val"

        if step is not None:
            self.fabric.print(f"Batch {step} | {eval_split_name} Loss: {loss:.4f}")
            self.fabric.log(f"{log_prefix}_loss", loss, step=step)
        else:
            self.fabric.print(f"{eval_split_name} Loss: {loss:.4f}")

        return loss

    def _batch_with_fasta(
        self, batch: Iterable[Sequence[GenomicInterval]]
    ) -> list[MemoryIntervalDataset]:
        results = list()

        for item in batch:
            result = MemoryIntervalDataset(item, self.config.fasta_path)
            results.append(result)

        return results
