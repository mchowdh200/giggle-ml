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
            all_labels = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .expand(batch_size, cluster_size)
            )
            all_labels = all_labels.reshape(-1)

            total_loss = 0.0
            total_correct = 0
            total_triplets = 0

            # Mine triplets for validation
            for i in range(len(all_embeds)):
                anchor_label = all_labels[i]
                anchor_embed = all_embeds[i]

                # Find positive and negative candidates
                positive_mask = (all_labels == anchor_label) & (
                    torch.arange(len(all_labels), device=self.device) != i
                )
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
                    all_embeds[negative_idx].unsqueeze(0),
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