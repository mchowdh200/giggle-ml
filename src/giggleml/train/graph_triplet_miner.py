# graph_triplet_miner.py

import torch
import torch.distributed as dist


def _euclidean_distance_matrix(
    embeds_x: torch.Tensor, embeds_y: torch.Tensor, safe_compute: bool = True
) -> torch.Tensor:
    """
    Compute pairwise **squared** L2 distances between embeds_x and embeds_y.

    Returns:
        Tensor of shape [len(embeds_x), len(embeds_y)] containing squared L2 distances.
    Notes:
        - We intentionally return squared distances (no sqrt). Squared distances
          preserve ordering (monotonic), are cheaper to compute, and are preferred
          for triplet mining comparisons. The `margin` must be interpreted in the
          same units (squared units) when using euclidean distances.
        - For numeric stability on mixed precision, if `safe_compute` is True we
          perform the heavy dot-product work in float32 and cast back to the
          input dtype.
    """
    # embeds_x: [local_N, D]
    # embeds_y: [N, D]
    if embeds_x.shape[-1] != embeds_y.shape[-1]:
        raise ValueError(
            "Embedding dimensionality must match between embeds_x and embeds_y"
        )

    orig_dtype = embeds_x.dtype
    compute_in_float = (
        safe_compute
        and torch.is_floating_point(embeds_x)
        and embeds_x.dtype != torch.float32
    )

    if compute_in_float:
        ex = embeds_x.float()
        ey = embeds_y.float()
    else:
        ex = embeds_x
        ey = embeds_y

    # squared norms
    x_norm_sq = torch.sum(ex * ex, dim=1, keepdim=True)  # [local_N, 1]
    y_norm_sq = torch.sum(ey * ey, dim=1, keepdim=True).T  # [1, N]

    # dot product
    dot_prod = torch.matmul(ex, ey.T)  # [local_N, N]

    # squared L2 distances
    dist_sq = x_norm_sq - 2.0 * dot_prod + y_norm_sq

    # numerical safety
    dist_sq = torch.clamp_min(dist_sq, 0.0)

    if compute_in_float and orig_dtype != torch.float32:
        # cast back to original dtype for downstream consistency
        dist_sq = dist_sq.to(orig_dtype)

    return dist_sq


class GraphTripletMiner:
    """
    Mines hard triplets from embeddings using a graph adjacency matrix.

    Design notes:
    - This miner assumes *every* rank has an identical copy of `embeddings`
      and `adjacency_matrix`. Each rank mines triplets for a non-overlapping
      slice of anchors. No all_gather of triplets is performed by default.
    - `margin` is interpreted in the distance space of the chosen `distance_metric`.
        * If distance_metric == 'euclidean' (default), this miner uses squared L2
          distances and the margin must be in squared-distance units.
        * If distance_metric == 'cosine', margin is in cosine-distance units
          (cosine-distance = 1 - cosine_similarity, in range ~[0, 2]).
    - If you want ordinary (not squared) L2 distances, adjust the code and
      margin interpretation accordingly (we prefer squared distances for speed).
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "euclidean",
        safe_fp_compute: bool = True,
    ):
        """
        Args:
            margin (float): Triplet margin in the chosen distance space.
            distance_metric (str): 'euclidean' (squared L2) or 'cosine'.
            safe_fp_compute (bool): Compute dot-products/norms in float32 for
                numeric stability when inputs are lower precision (fp16).
        """
        self.margin = margin
        self.distance_metric = distance_metric.lower()
        self.safe_fp_compute = safe_fp_compute

        if self.distance_metric == "euclidean":
            # Note: returns squared distances
            self.pdist_func = lambda x, y: _euclidean_distance_matrix(
                x, y, safe_compute=self.safe_fp_compute
            )
        elif self.distance_metric == "cosine":
            self.pdist_func = self._cosine_distance_matrix
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

    def _cosine_distance_matrix(
        self, embeds_x: torch.Tensor, embeds_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes pairwise cosine distance: 1.0 - cosine_similarity.

        Returns:
            Tensor [len(embeds_x), len(embeds_y)] with values typically in [0, 2].
        Notes:
            - We compute norms in float32 if inputs are lower-precision to improve stability.
            - The returned distance is **not** squared; it is 1 - cosine_similarity.
        """
        # cast to float for stable norms if requested
        compute_in_float = (
            self.safe_fp_compute
            and torch.is_floating_point(embeds_x)
            and embeds_x.dtype != torch.float32
        )

        if compute_in_float:
            x = embeds_x.float()
            y = embeds_y.float()
        else:
            x = embeds_x
            y = embeds_y

        # normalize
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        normalized_x = x / x_norm
        normalized_y = y / y_norm

        sim_matrix = torch.matmul(normalized_x, normalized_y.T)
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

        dist = 1.0 - sim_matrix  # cosine distance
        if compute_in_float and embeds_x.dtype != torch.float32:
            dist = dist.to(embeds_x.dtype)

        return dist

    def _get_local_slice(
        self, num_nodes: int, world_size: int, rank: int
    ) -> tuple[int, int]:
        """Calculate start index and local number of nodes for this rank (block distribution)."""
        chunk_size = num_nodes // world_size
        remainder = num_nodes % world_size

        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
        local_num_nodes = end_idx - start_idx

        return start_idx, local_num_nodes

    def mine(
        self, embeddings: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard triplets for anchors in this rank's slice.

        Args:
            embeddings (torch.Tensor): [num_nodes, embed_dim], identical across ranks.
            adjacency_matrix (torch.Tensor): [num_nodes, num_nodes], torch.bool,
                                             identical across ranks. adjacency[i,j] == True
                                             indicates j is a positive for i.

        Returns:
            (anchors_global_idx, positives_global_idx, negatives_global_idx)
            Each is a 1-D torch.long tensor on the same device as embeddings. They
            have equal length L (number of violating triplets found locally).
            If none found, all are empty tensors of dtype torch.long on `device`.
        """
        device = embeddings.device
        num_nodes = embeddings.shape[0]

        if adjacency_matrix.device != device:
            raise ValueError(
                "adjacency_matrix must be on the same device as embeddings"
            )
        if adjacency_matrix.dtype != torch.bool:
            raise ValueError("adjacency_matrix must be a torch.bool tensor")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be shape [num_nodes, embed_dim]")

        # 1) world size & rank
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 2) determine local slice
        start_idx, local_num_nodes = self._get_local_slice(num_nodes, world_size, rank)

        if local_num_nodes == 0:
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # 3) local embeddings slice (anchors)
        local_embeddings = embeddings[start_idx : start_idx + local_num_nodes]

        # 4) compute local distance matrix: [local_N, N]
        local_dist_matrix = self.pdist_func(local_embeddings, embeddings)

        # 5) local masks
        local_pos_mask = adjacency_matrix[
            start_idx : start_idx + local_num_nodes, :
        ]  # [local_N, N]
        local_neg_mask = ~local_pos_mask

        # explicit identity mask: True where candidate corresponds to self (global index)
        local_identity_mask = torch.zeros(
            (local_num_nodes, num_nodes), dtype=torch.bool, device=device
        )
        row_idx = torch.arange(local_num_nodes, device=device)
        col_idx = row_idx + start_idx
        local_identity_mask[row_idx, col_idx] = True

        local_valid_pos_mask = local_pos_mask & ~local_identity_mask
        local_valid_neg_mask = local_neg_mask & ~local_identity_mask

        # 6) find anchors that have at least one pos and one neg
        has_valid_pos = local_valid_pos_mask.any(dim=1)
        has_valid_neg = local_valid_neg_mask.any(dim=1)
        valid_anchor_mask_local = has_valid_pos & has_valid_neg  # [local_N]

        if not valid_anchor_mask_local.any():
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # 7) find hardest positives and negatives for each anchor (in our slice)
        # For positives: we want the **largest** distance among positives (hardest positive)
        pos_dists = torch.where(local_valid_pos_mask, local_dist_matrix, -torch.inf)
        hard_pos_dist, hard_pos_idx = torch.max(pos_dists, dim=1)

        # For negatives: we want the **smallest** distance among negatives (hardest negative)
        neg_dists = torch.where(local_valid_neg_mask, local_dist_matrix, torch.inf)
        hard_neg_dist, hard_neg_idx = torch.min(neg_dists, dim=1)

        # 8) filter to valid anchors
        anchor_idx_all_local = torch.arange(local_num_nodes, device=device)
        anchor_idx_valid = anchor_idx_all_local[valid_anchor_mask_local]
        pos_idx_valid = hard_pos_idx[valid_anchor_mask_local]  # global indices (0..N-1)
        neg_idx_valid = hard_neg_idx[valid_anchor_mask_local]  # global indices

        d_ap = hard_pos_dist[valid_anchor_mask_local]
        d_an = hard_neg_dist[valid_anchor_mask_local]

        # 9) margin violation check: d_ap - d_an + margin > 0
        # Note: margin must be in same distance space as self.distance_metric.
        triplet_violation = d_ap - d_an + self.margin
        violating_mask_local = triplet_violation > 0

        if not violating_mask_local.any():
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        local_a = anchor_idx_valid[
            violating_mask_local
        ]  # local indices relative to slice
        local_p = pos_idx_valid[violating_mask_local]  # global indices
        local_n = neg_idx_valid[violating_mask_local]  # global indices

        anchors_global = local_a + start_idx

        # ensure returned tensors are long and on the correct device
        anchors_global = anchors_global.to(dtype=torch.long, device=device)
        local_p = local_p.to(dtype=torch.long, device=device)
        local_n = local_n.to(dtype=torch.long, device=device)

        return anchors_global, local_p, local_n
