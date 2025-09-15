"""
Main training orchestrator for HyenaDNA fine-tuning with seqpare similarity metrics.
"""

import argparse
from pathlib import Path
from typing import cast

from lightning_fabric import Fabric
from lightning_fabric.loggers.tensorboard import TensorBoardLogger
from lightning_fabric.wrappers import nn
from torch import Tensor, optim
from torch.nn.modules.loss import TripletMarginLoss

from giggleml.embed_gen.embed_model import EmbedModel, HyenaDNA
from giggleml.embed_gen.embed_pipeline import DirectPipeline
from giggleml.train.embed_utils import embed_batch
from giggleml.train.rme_clusters_dataset import RmeSeqpareClusters
from giggleml.train.seqpare_db import SeqpareDB
from giggleml.train.triplet_trainer import IntervalClusterTripletFT
from giggleml.utils.cv_splits import create_cv_splits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HyenaDNA fine-tuning with seqpare")
    parser.add_argument(
        "--use_cv", action="store_true", help="Use cross-validation splits"
    )
    parser.add_argument(
        "--cv_split",
        choices=["train", "val", "test"],
        default="train",
        help="Which CV split to use",
    )

    # Core hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate override")
    parser.add_argument("--margin", type=float, help="Triplet margin override")
    parser.add_argument(
        "--clusters_per_batch", type=int, help="Clusters per batch override"
    )
    parser.add_argument(
        "--cluster_size", type=int, help="Intervals per cluster override"
    )
    parser.add_argument("--density", type=int, help="Intervals per candidate override")
    parser.add_argument(
        "--positive_threshold", type=int, help="Seqpare positive threshold override"
    )
    parser.add_argument("--epochs", type=int, help="Training epochs override")

    # AdamW hyperparameters
    parser.add_argument("--beta1", type=float, help="AdamW beta1 override")
    parser.add_argument("--beta2", type=float, help="AdamW beta2 override")
    parser.add_argument(
        "--weight_decay", type=float, help="AdamW weight decay override"
    )

    # Training control
    parser.add_argument(
        "--validation_freq", type=int, default=5, help="Validation frequency (epochs)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume_from_epoch", type=int, help="Resume training from specific epoch"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # INFO: ---------------------------
    #       Config
    # ---------------------------------

    # paths
    rme_dir = Path("data/roadmap_epigenomics")
    rme_beds = Path(rme_dir, "beds")
    seqpare_dir = Path(rme_dir, "seqpareRanks")
    training_dir = Path("models/hdna_seqpare_08092025")
    log_dir = Path(training_dir, "logs")
    checkpoint_dir = Path(training_dir, "checkpoints")
    fasta = Path("data/hg/hg19.fa")

    # Validate required paths exist
    if not rme_beds.exists():
        raise FileNotFoundError(f"Roadmap epigenomics directory not found: {rme_beds}")
    if not seqpare_dir.exists():
        raise FileNotFoundError(f"Seqpare directory not found: {seqpare_dir}")
    if not fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta}")

    # Cross-validation splits
    allowed_rme_names = None
    val_rme_names = None
    if args.use_cv:
        cv_ratio = {"train_ratio": 0.89, "val_ratio": 0.01, "test_ratio": 0.1}
        cv_splits = create_cv_splits(rme_beds, **cv_ratio)
        allowed_rme_names = cv_splits[args.cv_split]
        if args.cv_split == "train":
            val_rme_names = cv_splits["val"]
        print(
            f"Using CV split '{args.cv_split}' with {len(allowed_rme_names)} RME files"
        )
        if val_rme_names:
            print(f"Validation set has {len(val_rme_names)} RME files")

    # cluster sampling (with overrides)
    clusters_per_batch = args.clusters_per_batch or 10
    cluster_size = args.cluster_size or 10
    centroid_size = args.density or 30

    # training hyperparameters (with overrides)
    epochs = args.epochs or 10
    learning_rate = args.learning_rate or 1e-7
    margin = args.margin or 3.0

    # AdamW hyperparameters (with overrides)
    beta1 = args.beta1 or 0.9
    beta2 = args.beta2 or 0.999
    weight_decay = args.weight_decay or 0.0

    emodel: EmbedModel = HyenaDNA("1k", training=True)
    optimizer = optim.AdamW(
        lr=learning_rate,
        params=emodel.trainable_model.parameters(),
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    loss = TripletMarginLoss(margin=margin)

    # the embed pipeline is used for inference
    pipeline = DirectPipeline(emodel, 1)

    # other
    seqpare_positive_threshold = args.positive_threshold or 0.7

    # INFO: ----------------------------
    #       Fabric Setup
    # ---------------------------------
    # Use DDP strategy for multi-GPU training.
    # 'auto' will intelligently select the accelerator (CUDA, MPS, CPU) and devices.
    logger = TensorBoardLogger(root_dir=log_dir)
    fabric = Fabric(
        accelerator="auto", strategy="auto", devices="auto", loggers=[logger]
    )
    fabric.launch()  # Entry point for distributed training
    world_size = fabric.world_size
    rank = fabric.global_rank
    device = fabric.device

    # INFO: ---------------------------
    #       Setup Model, Optimizer, and Data
    # ---------------------------------

    seqpare_db = SeqpareDB(seqpare_dir)

    # Calculate start iteration for resumption
    start_iteration = 0
    if args.resume_from_epoch:
        start_iteration = args.resume_from_epoch
        print(f"Resuming from epoch {args.resume_from_epoch}")

    dataset = RmeSeqpareClusters(
        road_epig_path=rme_beds,
        seqpare=seqpare_db,
        world_size=world_size,
        rank=rank,
        positive_threshold=seqpare_positive_threshold,
        clusters_amnt=clusters_per_batch * world_size,
        groups_per_cluster=cluster_size,
        intervals_per_group=centroid_size,
        allowed_rme_names=allowed_rme_names,
        seed=args.seed,
    )

    # Create validation dataset if needed
    val_dataset = None
    if val_rme_names:
        val_dataset = RmeSeqpareClusters(
            road_epig_path=rme_beds,
            seqpare=seqpare_db,
            world_size=world_size,
            rank=rank,
            positive_threshold=seqpare_positive_threshold,
            clusters_amnt=max(5, clusters_per_batch // 2) * world_size,
            groups_per_cluster=cluster_size,
            intervals_per_group=centroid_size,
            allowed_rme_names=val_rme_names,
            seed=args.seed + 1,  # Different seed for validation
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
        dataset = RmeSeqpareClusters.load_dict(
            state["train_dataset_state"],
            seqpare_db,
        )
        if val_dataset and state["val_dataset_state"]:
            val_dataset = RmeSeqpareClusters.load_dict(
                state["val_dataset_state"],
                seqpare_db,
            )
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
        batch_tensor = batch_tensor.reshape(
            clusters_per_batch * world_size, cluster_size, emodel.embed_dim
        ).to(device)

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
                val_batch_tensor = embed_batch(
                    pipeline, emodel.embed_dim, val_batch, fasta
                )
                val_batch_tensor: Tensor = cast(
                    Tensor, fabric.all_gather(val_batch_tensor)
                )
                val_batch_tensor = val_batch_tensor.reshape(
                    -1, *val_batch_tensor.shape[2:]
                ).to(device)

                val_loss, val_accuracy = train_step.validation_step(val_batch_tensor)
                fabric.print(
                    f"Epoch {epoch} / {epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
                )
                fabric.log("val_loss", val_loss, step=epoch)
                fabric.log("val_accuracy", val_accuracy, step=epoch)
            except StopIteration:
                fabric.print("Validation data exhausted, skipping validation")

        # INFO: ---------------------------
        #       Save Checkpoint
        # ---------------------------------
        # Fabric ensures this only happens on the main process to prevent race conditions.
        state = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "train_dataset_state": dataset.save_dict(),
            "val_dataset_state": val_dataset.save_dict() if val_dataset else None,
        }
        fabric.save(checkpoint_path, state)
        fabric.print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    fabric.print("Training finished!")


if __name__ == "__main__":
    main()
