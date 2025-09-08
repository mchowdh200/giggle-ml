from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from random import Random
from typing import cast, override

import numpy as np
import torch
from lightning_fabric import Fabric
from lightning_fabric.loggers.tensorboard import TensorBoardLogger
from lightning_fabric.wrappers import nn
from numpy._typing import NDArray
from torch import Tensor, optim
from torch.nn import Module
from torch.nn.modules.loss import TripletMarginLoss
from torch.types import Device
from torch.utils.data import IterableDataset

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import (
    BedDataset,
    IntervalDataset,
    MemoryIntervalDataset,
)
from giggleml.embed_gen.embed_io import Embed
from giggleml.embed_gen.embed_model import EmbedModel, HyenaDNA, TrainableEmbedModel
from giggleml.embed_gen.embed_pipeline import DirectPipeline, EmbedPipeline
from giggleml.utils.misc import partition_integer, partition_list
from giggleml.utils.path_utils import as_path, fix_bed_ext
from giggleml.utils.types import GenomicInterval


class SeqpareDB:
    """reads all seqpare form .tsv files in the directory, taking the name, without last suffix, as the label"""

    def __init__(self, dir: str | Path, positive_threshold: float = 0.7) -> None:
        """categorizes into positive & negative based on the similarity"""
        self.dir: Path = as_path(dir)
        self.positive_threshold: float = positive_threshold
        self._labels: dict[str, int] = dict()

        for file in self.dir.iterdir():
            if file.suffix == ".tsv":
                self._labels[file.stem] = len(self._labels)

    @cache
    def _fetch_dense(self, label: str) -> NDArray[np.bool]:
        """@returns (positives, negatives) for a label"""

        path = self.dir / (label + ".bed.tsv")

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r") as f:
            next(f)
            bits: NDArray[np.bool] = np.zeros(len(self._labels), dtype=np.bool)

            for line in f:
                # parse the seqpare tsv file
                terms = line.split()
                # the column that corresponds to file names
                item = terms[5]
                # these are in the form ./label.bed
                item_id = self._labels[item[2:]]
                positive = float(terms[4]) >= self.positive_threshold
                bits[item_id] = positive

            return bits

    def fetch(self, label: str) -> tuple[list[str], list[str]]:
        """@returns (positives, negatives) for a label"""
        bits = self._fetch_dense(label)
        positives, negatives = list(), list()
        labels = list(self._labels.keys())

        for i, bit in enumerate(bits):
            label = labels[i]

            if bit:
                positives.append(label)
            else:
                negatives.append(label)

        return positives, negatives


# A cluster is a list of bed files
Cluster = list[list[GenomicInterval]]


# The "RME" (roadmap epigenomics) dataset is a series of bed files with known names corresponding to
# tissues and chromatin states. Since the dataset is fixed, we iterate it by searching for known
# items, but we still need the directory path.
class RmeSeqpareClusters(IterableDataset):
    """The dataset yields a series of interval clusters where only intervals within the same cluster should be considered positive."""

    def __init__(
        self,
        road_epig_path: str | Path,
        seqpare: SeqpareDB,
        world_size: int,
        rank: int,
        clusters_amnt: int = 10,
        cluster_size: int = 10,
        density: int = 30,
    ) -> None:
        """
        @param density: the amount of intervals per candidate
        """
        super().__init__()
        self.road_epig_path: Path = as_path(road_epig_path)
        self.seqpare_db: SeqpareDB = seqpare
        self.world_size: int = world_size
        self.rank: int = rank
        self.anchors: int = clusters_amnt
        self.cluster_size: int = cluster_size
        self.density: int = density
        self._rng: Random = Random(42)

    @override
    def __iter__(self) -> Iterator[list[Cluster]]:
        # "my" -- this rank
        my_rng: Random = Random(self.rank)
        # an identical value across the world size
        anchors = self._rng.choices(rme.bed_names, k=self.anchors)
        my_anchors = partition_list(anchors, self.world_size, self.rank)
        my_clusters = list[Cluster]()

        for anchor in my_anchors:
            neighbors, _ = self.seqpare_db.fetch(anchor)
            # in the event that an anchor has little neighbors, this will repeatedly resample
            # from the same labels
            labels = my_rng.choices([anchor, *neighbors], k=self.cluster_size)
            cluster: Cluster = list()

            # map into interval list
            for label in labels:
                path = fix_bed_ext(self.road_epig_path / label)
                bed = list(iter(BedDataset(path)))
                intervals = my_rng.choices(bed, k=self.density)
                cluster.append(intervals)

            my_clusters.append(cluster)

        yield my_clusters


# TODO: where's the cross validation?


EmbedCluster = list[Embed]
Loss3 = Callable[[Tensor, Tensor, Tensor], Tensor]


class IntervalClusterTripletFT(Module):
    def __init__(
        self,
        world_size: int,
        rank: int,
        device: Device,
        model: TrainableEmbedModel,
        loss: Loss3,
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.model: TrainableEmbedModel = model
        self.loss: Loss3 = loss

    def training_step(self, batch: Tensor):
        """
        Assumes that the batch is identical across all ranks.
        The separation of labor happens internally.
        @param batch: shape[cluster_amnt, cluster_size, embed_dim]
        """
        # basics
        edim = self.model.embed_dim
        rank = self.rank
        world_size = self.world_size

        # online hard triplet mining:
        # We're looking for within a category, the furthest from anchor
        # and outside the category, the closest to the anchor

        all_embeds = batch.reshape(-1, edim)

        # this rank operates on a subset of the bach
        splits = partition_integer(len(batch), world_size)
        # "my" means this rank
        my_start_cluster = sum(splits[:rank])
        my_amnt_clusters = splits[rank]
        my_embeds = batch[my_start_cluster : my_start_cluster + my_amnt_clusters].view(
            -1, edim
        )

        # INFO:
        # high vram cost.
        # using L2 distance.
        dist_matrix = torch.cdist(my_embeds, all_embeds, p=2.0)

        # Create a mask to identify positive pairs (anchor and other have the same label)
        all_labels = (
            torch.arange(batch.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(batch.shape[0], batch.shape[1])
        )
        my_labels = all_labels[my_start_cluster : my_start_cluster + my_amnt_clusters]
        positive_mask = (my_labels.reshape(-1).unsqueeze(1)) == (
            all_labels.reshape(-1).unsqueeze(0)
        )

        # positives with max dist
        pos_dist = dist_matrix.clone()
        pos_dist[~positive_mask] = -1.0
        _, pos_indices = torch.max(pos_dist, dim=1)
        positives = all_embeds[pos_indices]

        # negatives with min dist
        neg_dist = dist_matrix.clone()
        neg_dist[positive_mask] = float("inf")
        _, neg_indices = torch.min(neg_dist, dim=1)
        negatives = all_embeds[neg_indices]

        # hard triplets anchored by this rank's embeds
        return self.loss(my_embeds, positives, negatives)


def embed_batch(
    pipeline: EmbedPipeline, edim: int, batch: list[Cluster], fasta: Path | str
):
    fasta = as_path(fasta)
    flat_input: list[IntervalDataset] = list()

    for cluster in batch:
        for intervals in cluster:
            dataset = MemoryIntervalDataset(intervals, fasta)
            flat_input.append(dataset)

    embeds = pipeline.embed(flat_input)
    return embeds.reshape(len(batch), -1, edim)


# ---------------------------------
#     Main Training Function
# ---------------------------------


def main():
    # INFO: ---------------------------
    #       Config
    # ---------------------------------

    # paths
    rme_dir = Path("data/roadmap_epigenomics/beds")
    seqpare_dir = Path("data/roadmap_epigenomics/seqpareRanks")
    training_dir = Path("models/hdna_seqpare_08092025")
    log_dir = Path(training_dir, "logs")
    checkpoint_dir = Path(training_dir, "checkpoints")
    fasta = Path("data/hg/hg19.fa")

    # cluster sampling
    clusters_per_batch = 10
    cluster_size = 10
    centroid_size = 30

    # training
    epochs = 10
    emodel: EmbedModel = HyenaDNA("1k", training=True)
    optimizer = optim.AdamW(lr=1e-7, params=emodel.trainable_model.parameters())
    loss = TripletMarginLoss(margin=3)

    # the embed pipeline is used for inference
    pipeline = DirectPipeline(emodel, 64)

    # other
    seqpare_positive_threshold = 0.7

    # INFO: ----------------------------
    #       Fabric Setup
    # ---------------------------------
    # Use DDP strategy for multi-GPU training.
    # 'auto' will intelligently select the accelerator (CUDA, MPS, CPU) and devices.
    logger = TensorBoardLogger(root_dir=log_dir)
    fabric = Fabric(
        accelerator="auto", strategy="ddp", devices="auto", loggers=[logger]
    )
    fabric.launch()  # Entry point for distributed training
    world_size = fabric.world_size
    rank = fabric.global_rank
    device = fabric.device

    # INFO: ---------------------------
    #       Setup Model, Optimizer, and Data
    # ---------------------------------

    seqpare_db = SeqpareDB(seqpare_dir, seqpare_positive_threshold)
    dataset = RmeSeqpareClusters(
        rme_dir,
        seqpare_db,
        world_size,
        rank,
        clusters_per_batch,
        cluster_size,
        centroid_size,
    )
    train_step = IntervalClusterTripletFT(world_size, rank, device, emodel, loss)
    # Use fabric.setup to wrap the components. This prepares them for the
    # chosen hardware and strategy (e.g., wraps model in DDP).
    model: nn.Module = emodel.trainable_model
    model, optimizer = fabric.setup(model, optimizer)

    # INFO: ----------------------------
    #       Prepare for Resumption
    # ---------------------------------
    # Create directories and define the single checkpoint file path.
    if fabric.is_global_zero:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest-checkpoint.pt"

    start_epoch = 0
    if fabric.is_global_zero and checkpoint_path.is_file():
        fabric.print(f"Resuming from checkpoint: {checkpoint_path}")
        # `fabric.load` safely loads the checkpoint ONLY on the main process
        state = fabric.load(checkpoint_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1

    # Broadcast the start_epoch to all processes to ensure they are in sync
    start_epoch = fabric.broadcast(start_epoch)

    # INFO: ---------------------------
    #       Training Loop
    # ---------------------------------
    data = iter(dataset)

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        batch = next(data)
        batch_tensor = embed_batch(pipeline, emodel.embed_dim, batch, fasta)
        batch_tensor: Tensor = cast(Tensor, fabric.all_gather(batch_tensor))
        batch_tensor = batch_tensor.reshape(-1, *batch_tensor.shape[2:]).to(device)

        loss = train_step.training_step(batch_tensor)
        fabric.backward(loss)
        optimizer.step()

        fabric.print(f"Epoch {epoch} / {epochs} | Loss: {loss.item():.4f}")
        fabric.log("train_loss", loss.item(), step=epoch)

        # INFO: ---------------------------
        #       Save Checkpoint
        # ---------------------------------
        # Fabric ensures this only happens on the main process to prevent race conditions.
        state = {"model": model, "optimizer": optimizer, "epoch": epoch}
        fabric.save(checkpoint_path, state)
        fabric.print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    fabric.print("Training finished!")


if __name__ == "__main__":
    main()
