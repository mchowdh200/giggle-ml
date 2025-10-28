import torch
import torch.distributed as dist



def _euclidean_distance_matrix(
    embeds_x: torch.Tensor, embeds_y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the pairwise L2 distance matrix using matrix multiplication.
    This is more friendly to fp16 on CUDA than torch.cdist.
    """
    # embeds_x: [local_N, D]
    # embeds_y: [N, D]

    # Calculate squared norms
    # x_norm_sq: [local_N, 1]
    x_norm_sq = torch.sum(embeds_x**2, dim=1, keepdim=True)

    # y_norm_sq: [1, N]
    y_norm_sq = torch.sum(embeds_y**2, dim=1, keepdim=True).T

    # Calculate dot product
    # dot_prod: [local_N, N]
    dot_prod = torch.matmul(embeds_x, embeds_y.T)

    # Calculate squared L2 distance
    # Broadcasting handles the addition:
    # [local_N, 1] - [local_N, N] + [1, N] = [local_N, N]
    dist_sq = x_norm_sq - 2 * dot_prod + y_norm_sq

    # Clamp for numerical stability (floating point errors can -> small negatives)
    dist_sq.clamp_min_(0.0)

    # Return L2 distance
    return torch.sqrt(dist_sq)


class GraphTripletMiner:
    """
    Mines hard triplets from embeddings based on a graph adjacency matrix.

    This implementation is designed for distributed (DDP) environments
    but functions correctly in a single-process context.

    Each rank processes a unique slice of the anchors and returns
    only the triplets it found, without a final all_gather.
    This allows for calculating a local loss on each rank.
    """

    def __init__(self, margin: float = 1.0, distance_metric: str = "euclidean"):
        """
        Args:
            margin (float): The margin to use for triplet selection.
            distance_metric (str): 'euclidean' (L2) or 'cosine'.
        """
        self.margin = margin
        self.distance_metric = distance_metric

        if distance_metric == "euclidean":
            # cdist computes the matrix of all-pairs L2-distances
            self.pdist_func = _euclidean_distance_matrix
        elif distance_metric == "cosine":
            # Cosine distance = 1.0 - cosine_similarity
            self.pdist_func = self._cosine_distance_matrix
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

    def _cosine_distance_matrix(
        self, embeds_x: torch.Tensor, embeds_y: torch.Tensor
    ) -> torch.Tensor:
        """Computes the pairwise cosine distance matrix between two sets."""
        # Normalize embeddings to unit vectors
        norm_x = torch.norm(embeds_x, p=2, dim=1, keepdim=True)
        normalized_x = embeds_x / (norm_x + 1e-8)

        norm_y = torch.norm(embeds_y, p=2, dim=1, keepdim=True)
        normalized_y = embeds_y / (norm_y + 1e-8)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(normalized_x, normalized_y.T)
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

        return 1.0 - sim_matrix

    def _get_local_slice(self, num_nodes, world_size, rank):
        """Calculates the start index and size for the current rank."""
        chunk_size = num_nodes // world_size
        remainder = num_nodes % world_size

        # Distribute remainder
        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
        local_num_nodes = end_idx - start_idx

        return start_idx, local_num_nodes

    def mine(self, embeddings: torch.Tensor, adjacency_matrix: torch.Tensor):
        """
        Mines hard triplets for the local rank's slice of anchors.

        Args:
            embeddings (torch.Tensor): Shape [num_nodes, embed_dim].
                                       Must be identical on all ranks.
            adjacency_matrix (torch.Tensor): Shape [num_nodes, num_nodes].
                                             Must be a torch.bool tensor
                                             on the same device as embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - anchor_indices (local rank's results, global indices)
            - positive_indices (local rank's results, global indices)
            - negative_indices (local rank's results, global indices)
        """
        device = embeddings.device
        num_nodes = embeddings.shape[0]

        if adjacency_matrix.device != device:
            raise ValueError(
                "adjacency_matrix must be on the same device as embeddings"
            )
        if not adjacency_matrix.dtype == torch.bool:
            raise ValueError("adjacency_matrix must be a torch.bool tensor")

        # 1. Determine world size and rank
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 2. Determine this rank's slice of anchors
        start_idx, local_num_nodes = self._get_local_slice(num_nodes, world_size, rank)

        # Handle case where a rank might get 0 nodes
        if local_num_nodes == 0:
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # 3. Get local embeddings (our anchors)
        local_embeddings = embeddings[start_idx : start_idx + local_num_nodes]

        # 4. Calculate local slice of distance matrix: [local_N, N]
        local_dist_matrix = self.pdist_func(local_embeddings, embeddings)

        # 5. Create *local* masks directly to save memory
        # This avoids creating [N, N] tensors
        local_pos_mask = adjacency_matrix[start_idx : start_idx + local_num_nodes, :]
        local_neg_mask = ~local_pos_mask

        # Create the local slice of the identity matrix
        # This is a [local_N, N] tensor with True at (i, i + start_idx)
        local_identity_mask = torch.zeros_like(local_pos_mask)
        diag_indices = torch.arange(
            start_idx, start_idx + local_num_nodes, device=device
        )
        local_identity_mask[
            torch.arange(local_num_nodes, device=device), diag_indices
        ] = True

        local_valid_pos_mask = local_pos_mask & ~local_identity_mask
        local_valid_neg_mask = local_neg_mask & ~local_identity_mask

        # 6. Find valid anchors *within our slice*
        has_valid_pos = local_valid_pos_mask.any(dim=1)
        has_valid_neg = local_valid_neg_mask.any(dim=1)
        valid_anchor_mask_local = has_valid_pos & has_valid_neg  # [local_N]

        if not valid_anchor_mask_local.any():
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # --- Find Hardest Positives (for our slice) ---
        # Use torch.where for a memory-efficient masked selection
        pos_dists = torch.where(local_valid_pos_mask, local_dist_matrix, -torch.inf)
        hard_pos_dist, hard_pos_idx = torch.max(pos_dists, dim=1)

        # --- Find Hardest Negatives (for our slice) ---
        neg_dists = torch.where(local_valid_neg_mask, local_dist_matrix, torch.inf)
        hard_neg_dist, hard_neg_idx = torch.min(neg_dists, dim=1)

        # 7. Filter down locally
        # anchor_idx_all_local are indices relative to the local slice (0 to local_N-1)
        anchor_idx_all_local = torch.arange(local_num_nodes, device=device)

        anchor_idx_valid = anchor_idx_all_local[valid_anchor_mask_local]
        pos_idx_valid = hard_pos_idx[valid_anchor_mask_local]
        neg_idx_valid = hard_neg_idx[valid_anchor_mask_local]

        d_ap = hard_pos_dist[valid_anchor_mask_local]
        d_an = hard_neg_dist[valid_anchor_mask_local]

        # 8. Final local filter for margin violation
        triplet_loss = d_ap - d_an + self.margin
        violating_mask_local = triplet_loss > 0

        # Get final indices for this rank's triplets
        # local_a is relative to the slice (0 to local_N-1)
        local_a = anchor_idx_valid[violating_mask_local]
        # p and n are already global (0 to N-1)
        local_p = pos_idx_valid[violating_mask_local]
        local_n = neg_idx_valid[violating_mask_local]

        # 9. Convert local anchor indices to global indices and return
        return local_a + start_idx, local_p, local_n
