from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from random import Random
from typing import cast, override
import argparse

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
from giggleml.utils.cv_splits import create_cv_splits


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
                if len(terms) < 6:
                    continue  # skip malformed lines
                # the column that corresponds to file names
                item = terms[5]
                # these are in the form ./label.bed
                if not item.startswith("./") or len(item) < 3:
                    continue  # skip unexpected format
                label_name = item[2:]
                if label_name not in self._labels:
                    continue  # skip unknown labels
                item_id = self._labels[label_name]
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
        allowed_rme_names: list[str] | None = None,
    ) -> None:
        """
        @param density: the amount of intervals per candidate
        @param allowed_rme_names: List of RME names to restrict sampling to. If None, uses all available.
        """
        super().__init__()
        self.road_epig_path: Path = as_path(road_epig_path)
        self.seqpare_db: SeqpareDB = seqpare
        self.world_size: int = world_size
        self.rank: int = rank
        self.anchors: int = clusters_amnt
        self.cluster_size: int = cluster_size
        self.density: int = density
        self.allowed_rme_names: set[str] = set(allowed_rme_names) if allowed_rme_names else set(rme.bed_names)
        self._rng: Random = Random(42)

    @override
    def __iter__(self) -> Iterator[list[Cluster]]:
        # "my" -- this rank
        my_rng: Random = Random(self.rank)
        # Filter available bed names to only allowed ones
        available_names = [name for name in rme.bed_names if name in self.allowed_rme_names]
        if not available_names:
            raise ValueError("No available RME names after filtering")
        
        # an identical value across the world size
        anchors = self._rng.choices(available_names, k=self.anchors)
        my_anchors = partition_list(anchors, self.world_size, self.rank)
        my_clusters = list[Cluster]()

        for anchor in my_anchors:
            neighbors, _ = self.seqpare_db.fetch(anchor)
            # Filter neighbors to only allowed names
            allowed_neighbors = [n for n in neighbors if n in self.allowed_rme_names]
            # in the event that an anchor has little neighbors, this will repeatedly resample
            # from the same labels
            candidate_labels = [anchor] + allowed_neighbors
            labels = my_rng.choices(candidate_labels, k=self.cluster_size)
            cluster: Cluster = list()

            # map into interval list
            for label in labels:
                path = fix_bed_ext(self.road_epig_path / label)
                if not path.exists():
                    raise FileNotFoundError(f"BED file not found: {path}")
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
    
    def validation_step(self, batch: Tensor) -> tuple[float, float]:
        """
        Run validation on a batch and return loss and triplet accuracy.
        Similar to training_step but with evaluation metrics.
        """
        # Set model to eval mode for validation
        self.model.trainable_model.eval()
        
        with torch.no_grad():
            # Use the same triplet mining logic as training
            edim = self.model.embed_dim
            all_embeds = batch.reshape(-1, edim)
            
            # For validation, use all data (don't split by rank)
            dist_matrix = torch.cdist(all_embeds, all_embeds, p=2.0)
            batch_size = batch.shape[0]
            cluster_size = batch.shape[1]
            
            # Create labels for clusters
            all_labels = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, cluster_size)
            all_labels = all_labels.reshape(-1)
            
            total_loss = 0.0
            total_correct = 0
            total_triplets = 0
            
            # Mine triplets for validation
            for i in range(len(all_embeds)):
                anchor_label = all_labels[i]
                anchor_embed = all_embeds[i]
                
                # Find positive and negative candidates
                positive_mask = (all_labels == anchor_label) & (torch.arange(len(all_labels), device=self.device) != i)
                negative_mask = all_labels != anchor_label
                
                if not positive_mask.any() or not negative_mask.any():
                    continue
                
                # Hard positive: furthest positive from anchor
                positive_dists = dist_matrix[i][positive_mask]
                hard_positive_idx = torch.argmax(positive_dists)
                positive_idx = torch.where(positive_mask)[0][hard_positive_idx]
                
                # Hard negative: closest negative to anchor  
                negative_dists = dist_matrix[i][negative_mask]
                hard_negative_idx = torch.argmin(negative_dists)
                negative_idx = torch.where(negative_mask)[0][hard_negative_idx]
                
                # Compute triplet loss
                triplet_loss = self.loss(
                    anchor_embed.unsqueeze(0),
                    all_embeds[positive_idx].unsqueeze(0),
                    all_embeds[negative_idx].unsqueeze(0)
                )
                total_loss += triplet_loss.item()
                
                # Compute triplet accuracy (d(a,p) < d(a,n))
                ap_dist = torch.norm(anchor_embed - all_embeds[positive_idx])
                an_dist = torch.norm(anchor_embed - all_embeds[negative_idx])
                if ap_dist < an_dist:
                    total_correct += 1
                
                total_triplets += 1
            
            # Compute averages
            avg_loss = total_loss / max(total_triplets, 1)
            accuracy = total_correct / max(total_triplets, 1)
        
        # Set model back to train mode
        self.model.trainable_model.train()
        
        return avg_loss, accuracy


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




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HyenaDNA fine-tuning with seqpare")
    parser.add_argument("--use_cv", action="store_true", help="Use cross-validation splits")
    parser.add_argument("--cv_split", choices=["train", "val", "test"], default="train", 
                       help="Which CV split to use")
    parser.add_argument("--learning_rate", type=float, help="Learning rate override")
    parser.add_argument("--margin", type=float, help="Triplet margin override") 
    parser.add_argument("--clusters_per_batch", type=int, help="Clusters per batch override")
    parser.add_argument("--validation_freq", type=int, default=5, help="Validation frequency (epochs)")
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    # Validate required paths exist
    if not rme_dir.exists():
        raise FileNotFoundError(f"Roadmap epigenomics directory not found: {rme_dir}")
    if not seqpare_dir.exists():
        raise FileNotFoundError(f"Seqpare directory not found: {seqpare_dir}")
    if not fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta}")

    # Cross-validation splits
    allowed_rme_names = None
    val_rme_names = None
    if args.use_cv:
        cv_splits = create_cv_splits(rme_dir)
        allowed_rme_names = cv_splits[args.cv_split]
        if args.cv_split == "train":
            val_rme_names = cv_splits["val"]
        print(f"Using CV split '{args.cv_split}' with {len(allowed_rme_names)} RME files")
        if val_rme_names:
            print(f"Validation set has {len(val_rme_names)} RME files")

    # cluster sampling
    clusters_per_batch = args.clusters_per_batch or 10
    cluster_size = 10
    centroid_size = 30

    # training hyperparameters (with overrides)
    epochs = 10
    learning_rate = args.learning_rate or 1e-7
    margin = args.margin or 3.0
    
    emodel: EmbedModel = HyenaDNA("1k", training=True)
    optimizer = optim.AdamW(lr=learning_rate, params=emodel.trainable_model.parameters())
    loss = TripletMarginLoss(margin=margin)

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
        allowed_rme_names,
    )
    
    # Create validation dataset if needed
    val_dataset = None
    if val_rme_names:
        val_dataset = RmeSeqpareClusters(
            rme_dir,
            seqpare_db,
            world_size,
            rank,
            clusters_per_batch,
            cluster_size,
            centroid_size,
            val_rme_names,
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
    if checkpoint_path.is_file():
        fabric.print(f"Resuming from checkpoint: {checkpoint_path}")
        # Load checkpoint on all ranks to ensure model/optimizer sync
        state = fabric.load(checkpoint_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1

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

        fabric.print(f"Epoch {epoch} / {epochs} | Train Loss: {loss.item():.4f}")
        fabric.log("train_loss", loss.item(), step=epoch)

        # Validation phase
        if val_dataset and (epoch + 1) % args.validation_freq == 0:
            try:
                val_data = iter(val_dataset)
                val_batch = next(val_data)
                val_batch_tensor = embed_batch(pipeline, emodel.embed_dim, val_batch, fasta)
                val_batch_tensor: Tensor = cast(Tensor, fabric.all_gather(val_batch_tensor))
                val_batch_tensor = val_batch_tensor.reshape(-1, *val_batch_tensor.shape[2:]).to(device)
                
                val_loss, val_accuracy = train_step.validation_step(val_batch_tensor)
                fabric.print(f"Epoch {epoch} / {epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
                fabric.log("val_loss", val_loss, step=epoch)
                fabric.log("val_accuracy", val_accuracy, step=epoch)
            except StopIteration:
                fabric.print("Validation data exhausted, skipping validation")

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
