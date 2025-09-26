"""
Triplet loss training logic for fine-tuning models with interval clusters.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.types import Device

from giggleml.embed_gen.embed_model import EmbedModel
from giggleml.utils.misc import partition_integer

# Type alias for triplet loss function
Loss3 = Callable[[Tensor, Tensor, Tensor], Tensor]


class IntervalClusterTripletFT(Module):
    def __init__(
        self,
        world_size: int,
        rank: int,
        device: Device,
        model: EmbedModel,
        loss: Loss3,
    ):
        super().__init__()
        self.world_size: int = world_size
        self.rank: int = rank
        self.device: Device = device
        self.model: EmbedModel = model
        self.loss: Loss3 = loss

    def _create_cluster_labels(self, batch: Tensor) -> Tensor:
        """Create labels for cluster assignment."""
        return (
            torch.arange(batch.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(batch.shape[0], batch.shape[1])
            .reshape(-1)
        )

    def _prepare_embeddings_and_labels(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Prepare embeddings and labels from batch."""
        all_embeds = batch.reshape(-1, self.model.embed_dim)
        all_labels = self._create_cluster_labels(batch)
        return all_embeds, all_labels

    def _mine_hard_triplets(
        self, embeds: Tensor, all_embeds: Tensor, labels: Tensor, all_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Mine hard triplets using online hard triplet mining.
        Returns hard positives and hard negatives for the given embeds.
        """
        dist_matrix = torch.cdist(embeds, all_embeds, p=2.0)

        positive_mask = (labels.unsqueeze(1)) == (all_labels.unsqueeze(0))

        # Hard positives: furthest positive from anchor
        pos_dist = dist_matrix.clone()
        pos_dist[~positive_mask] = -1.0
        _, pos_indices = torch.max(pos_dist, dim=1)
        positives = all_embeds[pos_indices]

        # Hard negatives: closest negative to anchor
        neg_dist = dist_matrix.clone()
        neg_dist[positive_mask] = float("inf")
        _, neg_indices = torch.min(neg_dist, dim=1)
        negatives = all_embeds[neg_indices]

        return positives, negatives

    def _get_rank_subset(
        self, batch: Tensor, all_embeds: Tensor, all_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Get this rank's subset of embeddings and labels from the batch."""
        splits = partition_integer(len(batch), self.world_size)
        my_start_cluster = sum(splits[: self.rank])
        my_amnt_clusters = splits[self.rank]
        my_embeds = batch[my_start_cluster : my_start_cluster + my_amnt_clusters].view(
            -1, self.model.embed_dim
        )
        my_labels = all_labels[
            my_start_cluster * batch.shape[1] : (my_start_cluster + my_amnt_clusters)
            * batch.shape[1]
        ]
        return my_embeds, my_labels

    def _compute_loss_and_accuracy(
        self, anchors: Tensor, positives: Tensor, negatives: Tensor
    ) -> tuple[float, float]:
        """Compute triplet loss and accuracy metrics."""
        triplet_losses = self.loss(anchors, positives, negatives)
        total_loss = triplet_losses.sum().item()

        # Compute triplet accuracy (d(a,p) < d(a,n))
        ap_dists = torch.norm(anchors - positives, dim=1)
        an_dists = torch.norm(anchors - negatives, dim=1)
        total_correct = (ap_dists < an_dists).sum().item()

        total_triplets = len(anchors)
        avg_loss = total_loss / max(total_triplets, 1)
        accuracy = total_correct / max(total_triplets, 1)

        return avg_loss, accuracy

    def training_step(self, batch: Tensor):
        """
        Assumes that the batch is identical across all ranks.
        @param batch: shape[cluster_amnt, cluster_size, embed_dim]
        """
        all_embeds, all_labels = self._prepare_embeddings_and_labels(batch)

        # This rank operates on a subset of the batch
        my_embeds, my_labels = self._get_rank_subset(batch, all_embeds, all_labels)

        # Mine hard triplets using the shared method
        positives, negatives = self._mine_hard_triplets(
            my_embeds, all_embeds, my_labels, all_labels
        )

        return self.loss(my_embeds, positives, negatives)

    def validation_step(self, batch: Tensor) -> tuple[float, float]:
        """
        Run validation on a batch and return loss and triplet accuracy.
        Uses the same distributed strategy as training_step.
        """
        self.model.eval()

        with torch.no_grad():
            all_embeds, all_labels = self._prepare_embeddings_and_labels(batch)

            # This rank operates on a subset of the batch (same as training_step)
            my_embeds, my_labels = self._get_rank_subset(batch, all_embeds, all_labels)

            # Mine hard triplets using the shared method
            positives, negatives = self._mine_hard_triplets(
                my_embeds, all_embeds, my_labels, all_labels
            )

            # Compute loss and accuracy using the shared method
            avg_loss, accuracy = self._compute_loss_and_accuracy(
                my_embeds, positives, negatives
            )

        self.model.train()
        return avg_loss, accuracy
