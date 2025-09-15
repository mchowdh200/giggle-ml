"""
Triplet loss training logic for fine-tuning models with interval clusters.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.types import Device

from giggleml.embed_gen.embed_model import TrainableEmbedModel
from giggleml.utils.misc import partition_integer

# Type alias for triplet loss function
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
        self.world_size: int = world_size
        self.rank: int = rank
        self.device: Device = device
        self.model: TrainableEmbedModel = model
        self.loss: Loss3 = loss

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

    def training_step(self, batch: Tensor):
        """
        Assumes that the batch is identical across all ranks.
        @param batch: shape[cluster_amnt, cluster_size, embed_dim]
        """
        edim = self.model.embed_dim
        all_embeds = batch.reshape(-1, edim)

        # This rank operates on a subset of the batch
        splits = partition_integer(len(batch), self.world_size)
        my_start_cluster = sum(splits[: self.rank])
        my_amnt_clusters = splits[self.rank]
        my_embeds = batch[my_start_cluster : my_start_cluster + my_amnt_clusters].view(
            -1, edim
        )

        # Create labels for all clusters
        all_labels = (
            torch.arange(batch.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(batch.shape[0], batch.shape[1])
            .reshape(-1)
        )
        my_labels = all_labels[
            my_start_cluster * batch.shape[1] : (my_start_cluster + my_amnt_clusters)
            * batch.shape[1]
        ]

        # Mine hard triplets using the shared method
        positives, negatives = self._mine_hard_triplets(
            my_embeds, all_embeds, my_labels, all_labels
        )

        return self.loss(my_embeds, positives, negatives)

    def validation_step(self, batch: Tensor) -> tuple[float, float]:
        """
        Run validation on a batch and return loss and triplet accuracy.
        Similar to training_step but with evaluation metrics.
        """
        self.model.trainable_model.eval()

        with torch.no_grad():
            edim = self.model.embed_dim
            all_embeds = batch.reshape(-1, edim)

            # Create labels for clusters
            all_labels = (
                torch.arange(batch.shape[0], device=self.device)
                .unsqueeze(1)
                .expand(batch.shape[0], batch.shape[1])
                .reshape(-1)
            )

            # Mine hard triplets using the shared method
            positives, negatives = self._mine_hard_triplets(
                all_embeds, all_embeds, all_labels, all_labels
            )

            total_loss = 0.0
            total_correct = 0
            total_triplets = len(all_embeds)

            # Compute triplet loss and accuracy
            triplet_losses = self.loss(all_embeds, positives, negatives)
            total_loss = triplet_losses.sum().item()

            # Compute triplet accuracy (d(a,p) < d(a,n))
            ap_dists = torch.norm(all_embeds - positives, dim=1)
            an_dists = torch.norm(all_embeds - negatives, dim=1)
            total_correct = (ap_dists < an_dists).sum().item()

            # Compute averages
            avg_loss = total_loss / max(total_triplets, 1)
            accuracy = total_correct / max(total_triplets, 1)

        self.model.trainable_model.train()
        return avg_loss, accuracy

