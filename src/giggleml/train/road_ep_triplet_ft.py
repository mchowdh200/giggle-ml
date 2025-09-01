from typing import Any, TypedDict, cast, final, override

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from giggleml.embed_gen.chunk_de_chunk import ChunkDeChunk
from giggleml.embed_gen.embed_model import TrainableEmbedModel

"""
Training Loop:

1. (Iterable) Dataset is asked by the PL Trainer to create a new batch.
  1. The dataset will then randomly choose a series of labels to focus on.
  2. Within each related bed file, it randomly samples a series of clusters (GenomicInterval lists).
  3. It embeds the entire batch, which may require multiple batch inference jobs. The result is a
  list of embeddings, one per interval. Embeddings within a cluster are averaged. The final result
  is as many embeddings as there are clusters.
  4. Using the GPU, it performs a series of KNN scans to find optimal triplets (online mining).
2. The loss is computed for the entire batch, average of all triplets.

This requires GPU operations before (and during) the PL Model's training_step()

"""


Cluster = list[str]


class Triplet[T](TypedDict):
    anchor: T
    positive: T
    negative: T


@final
class LitModule(pl.LightningModule):
    """
    A PyTorch Lightning module to fine-tune the HyenaDNA embedding model
    using triplet loss.
    """

    def __init__(
        self,
        model: TrainableEmbedModel,
        embed_batch_size: int,
        cluster_size: int,
        labels_per_batch: int,
        clusters_per_label: int,
        workers: int,
        learning_rate: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        margin: float = 1.0,
    ):
        """
        @param embedBatchSize: the amount of --clusters-- per embedding batch. Necessary because
        while the loss is averaged over the batch, it is too large to embed all at once.
        """

        super().__init__()

        self.model = model
        self.underlying_model = model.trainable_model  # expose to pytorch lightning

        self.embed_batch_size = embed_batch_size
        self.cluster_size = cluster_size
        self.labels_per_batch = labels_per_batch
        self.clusters_per_label = clusters_per_label
        self.loss = nn.TripletMarginLoss(margin=margin, reduction="mean")

        self.save_hyperparameters(ignore=["workers", "model"])
        self.hparams["backing_model_architecture"] = str(model)

    @override
    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        # INFO: preventing batches from transferring to the GPU so that they can
        # be processed further on CPU first
        return batch
        # return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @override
    def forward(self, batch: list[str]) -> Tensor:
        return self.model.embed(batch)

    def _step(self, batch: list[Cluster], batch_idx: int):
        # The batch is a set of Cluster lists where each corresponds to a bed file in road epi
        # so, clusters within the same list share the same label -- this is our definition of
        # a positive

        # INFO: 1. embed the entire batch: [N, D, C] -> [N, D, embedDim] by averaging embeds within a cluster

        e_dim = self.model.embed_dim
        local_n = self.labels_per_batch
        D = self.clusters_per_label
        C = self.cluster_size

        sequences: list[str] = [interval for cluster in batch for interval in cluster]

        chunk_de_chunk = ChunkDeChunk(self.model.max_seq_len)
        chunked_seqs = chunk_de_chunk.chunk(sequences)
        batches = [
            chunked_seqs[i : min(i + self.embed_batch_size, len(chunked_seqs))]
            for i in range(0, len(chunked_seqs), self.embed_batch_size)
        ]
        chunked_embeds = torch.cat([self(batch) for batch in batches])
        local_embeds = (
            chunk_de_chunk.dechunk(chunked_embeds).view(local_n, D, C, e_dim).mean(dim=2)
        )

        assert local_embeds.device == self.device

        if self.trainer.world_size > 1:
            gather = cast(Tensor, self.all_gather(local_embeds))
        else:
            gather = local_embeds

        global_n = local_n * self.trainer.world_size
        flat = gather.view(global_n * D, e_dim)

        # INFO: 2. mine for triplets

        # We're looking for within a category, the furthest from anchor
        # and outside the category, the closest to the anchor

        # The anchors are the embeddings from the current worker's batch
        anchors = local_embeds.view(local_n * D, e_dim)
        rank = self.trainer.global_rank
        dist_matrix = torch.cdist(anchors, flat, p=2.0)  # INFO: using L2 distance

        # Create a mask to identify positive pairs (anchor and other have the same label)
        global_labels = (
            torch.arange(global_n, device=self.device)
            .unsqueeze(1)
            .expand(global_n, D)
            .reshape(-1)
        )
        anchor_labels = global_labels[rank * local_n * D : (rank + 1) * local_n * D]
        positive_mask = anchor_labels.unsqueeze(1) == global_labels.unsqueeze(0)

        # positives
        dist_pos = dist_matrix.clone()
        dist_pos[~positive_mask] = -1.0
        _, pos_indices = torch.max(dist_pos, dim=1)
        positives = flat[pos_indices]

        # negatives
        dist_neg = dist_matrix.clone()
        dist_neg[positive_mask] = float("inf")
        _, neg_indices = torch.min(dist_neg, dim=1)
        negatives = flat[neg_indices]

        # INFO: 3. compute the average loss

        # Calculate (average loss)
        return self.loss(anchors, positives, negatives)

    @override
    def training_step(self, batch: list[Cluster], batch_idx: int):
        loss = self._step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    @override
    def test_step(self, batch: list[Cluster], batch_idx: int):
        loss = self._step(batch, batch_idx)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    @override
    def validation_step(self, batch: list[Cluster], batch_idx: int):
        loss = self._step(batch, batch_idx)
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    @override
    def configure_optimizers(self):
        # Use the AdamW optimizer, which is common for Transformer-based models
        return AdamW(self.parameters(), lr=self.hparams["learningRate"])

    # TODO: validation/test_step
