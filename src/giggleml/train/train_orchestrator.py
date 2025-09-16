"""
Main training orchestrator for HyenaDNA fine-tuning with seqpare similarity metrics.
"""

import argparse
import math
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
        "--mode",
        choices=["train", "val", "test"],
        default="train",
        help="Training mode: train (train on train/test on val with checkpointing), val (same as train but no checkpointing), test (test on test set using checkpoint)",
    )

    # Core hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate override")
    parser.add_argument("--margin", type=float, help="Triplet margin override")
    parser.add_argument("--batch_size", type=int, help="Total batch size override")
    parser.add_argument(
        "--pk_ratio", type=float, help="PK ratio for cluster calculation override"
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


def run_training(
    use_cv: bool = False,
    mode: str = "train",
    learning_rate: float | None = None,
    margin: float | None = None,
    batch_size: int | None = None,
    pk_ratio: float | None = None,
    density: int | None = None,
    positive_threshold: int | None = None,
    epochs: int | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    weight_decay: float | None = None,
    validation_freq: int = 5,
    seed: int = 42,
    resume_from_epoch: int | None = None,
) -> tuple[float, float]:
    """
    Run training/evaluation with given hyperparameters.

    The function behaves differently based on mode:
    - "train": Train on train split, evaluate on val split, save checkpoints
    - "val": Train on train split, evaluate on val split, no checkpointing (for hyperparameter optimization)
    - "test": Load checkpoint and evaluate on test split only

    Returns:
        (primary_loss, primary_accuracy) where:
        - For mode="train": (val_loss, val_accuracy) from training on train set
        - For mode="val": (val_loss, val_accuracy) from training on train set
        - For mode="test": (test_loss, test_accuracy) from evaluation only
    """
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

    # Cross-validation splits and mode determination
    train_rme_names = None
    eval_rme_names = None
    is_training_mode = mode in [
        "val",
        "train",
    ]  # Both val and train modes involve training
    is_test_only = mode == "test"
    save_checkpoints = mode == "train"  # Only save checkpoints in train mode

    if use_cv:
        cv_ratio = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
        cv_splits = create_cv_splits(rme_beds, **cv_ratio)

        if mode in ["train", "val"]:
            # Both train and val modes: train on train set, evaluate on val set
            train_rme_names = cv_splits["train"]
            eval_rme_names = cv_splits["val"]
            checkpoint_mode = (
                "with checkpointing" if save_checkpoints else "no checkpointing"
            )
            print(
                f"{mode.capitalize()} mode: Training on train split ({len(train_rme_names)} files), evaluating on val split ({len(eval_rme_names)} files), {checkpoint_mode}"
            )

        elif mode == "test":
            # Test mode: evaluate only on test set
            eval_rme_names = cv_splits["test"]
            print(
                f"Test mode: Evaluating on test split with {len(eval_rme_names)} RME files"
            )

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'")

    # Convert batch_size and pk_ratio to clusters_per_batch and cluster_size
    batch_size = batch_size or 100
    pk_ratio = pk_ratio or 1.0

    # Calculate cluster_amnt and cluster_size from batch_size and pk_ratio
    # cluster_amnt = sqrt(batch_size * pk_ratio)
    # cluster_size = cluster_amnt / pk_ratio
    cluster_amnt = int(math.sqrt(batch_size * pk_ratio))
    cluster_size = int(cluster_amnt / pk_ratio)

    # Ensure minimum values
    clusters_per_batch = max(1, cluster_amnt)
    cluster_size = max(1, cluster_size)

    centroid_size = density or 30

    # training hyperparameters (with overrides)
    epochs = epochs or 10
    learning_rate = learning_rate or 1e-7
    margin = margin or 3.0

    # AdamW hyperparameters (with overrides)
    beta1 = beta1 or 0.9
    beta2 = beta2 or 0.999
    weight_decay = weight_decay or 0.0

    emodel: EmbedModel = HyenaDNA("16k", training=True)
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
    seqpare_positive_threshold = positive_threshold or 0.7

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
    if resume_from_epoch:
        start_iteration = resume_from_epoch
        print(f"Resuming from epoch {resume_from_epoch}")

    # Create datasets based on mode
    train_dataset = None
    eval_dataset = None

    if is_training_mode:
        # Create training dataset
        train_dataset = RmeSeqpareClusters(
            road_epig_path=rme_beds,
            seqpare=seqpare_db,
            world_size=world_size,
            rank=rank,
            positive_threshold=seqpare_positive_threshold,
            clusters_amnt=clusters_per_batch * world_size,
            groups_per_cluster=cluster_size,
            intervals_per_group=centroid_size,
            allowed_rme_names=train_rme_names,
            seed=seed,
        )

    # Create evaluation dataset (used in all modes)
    # For val mode (hyperparameter search), use smaller evaluation batches to save memory
    eval_clusters_amnt = (
        2 * world_size
        if mode == "val"
        else max(5, clusters_per_batch // 2) * world_size
    )

    eval_dataset = RmeSeqpareClusters(
        road_epig_path=rme_beds,
        seqpare=seqpare_db,
        world_size=world_size,
        rank=rank,
        positive_threshold=seqpare_positive_threshold,
        clusters_amnt=eval_clusters_amnt,
        groups_per_cluster=cluster_size,
        intervals_per_group=centroid_size,
        allowed_rme_names=eval_rme_names,
        seed=seed + 1,  # Different seed for evaluation
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
    if checkpoint_path.is_file() and (is_training_mode or is_test_only):
        fabric.print(f"Loading checkpoint: {checkpoint_path}")
        # Load checkpoint on all ranks to ensure model/optimizer sync
        state = fabric.load(checkpoint_path)
        model.load_state_dict(state["model"])
        if is_training_mode:  # Only load optimizer state for training modes
            optimizer.load_state_dict(state["optimizer"])
            # Only load dataset state for train mode (not val mode) to avoid memory issues during hparam search
            if mode == "train":
                if train_dataset and state.get("train_dataset_state"):
                    train_dataset = RmeSeqpareClusters.load_dict(
                        state["train_dataset_state"],
                        seqpare_db,
                    )
                if eval_dataset and state.get("eval_dataset_state"):
                    eval_dataset = RmeSeqpareClusters.load_dict(
                        state["eval_dataset_state"],
                        seqpare_db,
                    )
                start_epoch = state["epoch"] + 1

    # INFO: ---------------------------
    #       Training/Evaluation Loop
    # ---------------------------------

    if is_training_mode:
        # Training mode: train on the training dataset
        train_data = iter(train_dataset)
        final_eval_loss = None
        final_eval_accuracy = None

        for epoch in range(start_epoch, epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()

            batch = next(train_data)
            batch_tensor = embed_batch(pipeline, emodel.embed_dim, batch, fasta)
            batch_tensor: Tensor = cast(Tensor, fabric.all_gather(batch_tensor))
            batch_tensor = batch_tensor.reshape(
                clusters_per_batch * world_size, cluster_size, emodel.embed_dim
            ).to(device)

            loss = train_step.training_step(batch_tensor)
            fabric.backward(loss)
            optimizer.step()

            train_loss = loss.item()
            fabric.print(f"Epoch {epoch} / {epochs} | Train Loss: {train_loss:.4f}")
            fabric.log("train_loss", train_loss, step=epoch)

            # Evaluation phase (on eval_dataset)
            if eval_dataset and (epoch + 1) % validation_freq == 0:
                try:
                    model.eval()
                    eval_data = iter(eval_dataset)
                    eval_batch = next(eval_data)
                    eval_batch_tensor = embed_batch(
                        pipeline, emodel.embed_dim, eval_batch, fasta
                    )
                    eval_batch_tensor: Tensor = cast(
                        Tensor, fabric.all_gather(eval_batch_tensor)
                    )
                    eval_batch_tensor = eval_batch_tensor.reshape(
                        eval_clusters_amnt, cluster_size, emodel.embed_dim
                    ).to(device)

                    eval_loss, eval_accuracy = train_step.validation_step(
                        eval_batch_tensor
                    )
                    final_eval_loss = eval_loss
                    final_eval_accuracy = eval_accuracy

                    eval_split_name = "Val" if mode == "val" else "Test"
                    fabric.print(
                        f"Epoch {epoch} / {epochs} | {eval_split_name} Loss: {eval_loss:.4f} | {eval_split_name} Acc: {eval_accuracy:.4f}"
                    )
                    fabric.log(f"{eval_split_name.lower()}_loss", eval_loss, step=epoch)
                    fabric.log(
                        f"{eval_split_name.lower()}_accuracy", eval_accuracy, step=epoch
                    )
                    model.train()  # Switch back to training mode
                except StopIteration:
                    fabric.print("Evaluation data exhausted, skipping evaluation")

            # INFO: ---------------------------
            #       Save Checkpoint
            # ---------------------------------
            # Only save checkpoints in train mode
            if save_checkpoints:
                # Fabric ensures this only happens on the main process to prevent race conditions.
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "epoch": epoch,
                    "train_dataset_state": train_dataset.save_dict()
                    if train_dataset
                    else None,
                    "eval_dataset_state": eval_dataset.save_dict()
                    if eval_dataset
                    else None,
                }
                fabric.save(checkpoint_path, state)
                fabric.print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

        fabric.print("Training finished!")

        # Ensure we have evaluation metrics for training mode
        if final_eval_loss is None or final_eval_accuracy is None:
            raise RuntimeError(
                "Evaluation metrics not available - evaluation may not have run"
            )

        return final_eval_loss, final_eval_accuracy

    else:
        # Test-only mode: evaluate on test set only
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Test mode requires a checkpoint file at {checkpoint_path}"
            )

        model.eval()
        eval_data = iter(eval_dataset)

        fabric.print("Evaluating on test split...")

        # Run evaluation on a single batch
        batch = next(eval_data)
        batch_tensor = embed_batch(pipeline, emodel.embed_dim, batch, fasta)
        batch_tensor: Tensor = cast(Tensor, fabric.all_gather(batch_tensor))
        batch_tensor = batch_tensor.reshape(
            eval_clusters_amnt, cluster_size, emodel.embed_dim
        ).to(device)

        eval_loss, eval_accuracy = train_step.validation_step(batch_tensor)
        fabric.print(f"Test Loss: {eval_loss:.4f} | Test Acc: {eval_accuracy:.4f}")

        return eval_loss, eval_accuracy


def main():
    args = parse_args()

    primary_loss, primary_accuracy = run_training(
        use_cv=args.use_cv,
        mode=args.mode,
        learning_rate=args.learning_rate,
        margin=args.margin,
        batch_size=args.batch_size,
        pk_ratio=args.pk_ratio,
        density=args.density,
        positive_threshold=args.positive_threshold,
        epochs=args.epochs,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        validation_freq=args.validation_freq,
        seed=args.seed,
        resume_from_epoch=args.resume_from_epoch,
    )

    print(f"Final results - Loss: {primary_loss:.4f}, Accuracy: {primary_accuracy:.4f}")


if __name__ == "__main__":
    main()
