import torch

from giggleml.utils.torch_utils import get_rank, get_world_size


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
    Mines hard, semi-hard, or all triplets from embeddings using a graph adjacency matrix.

    Design notes:
    - This miner assumes *every* rank has an identical copy of `embeddings`
      and `adjacency_matrix`. Each rank mines triplets for a non-overlapping
      slice of anchors. No all_gather of triplets is performed by default.
    - `margin` (loss margin) and `mining_margin` are interpreted in the
      distance space of the chosen `distance_metric` (squared L2 units).
    """

    def __init__(
        self,
        margin: float = 1.0,
        safe_fp_compute: bool = True,
        mining_strategy: str = "hard",
        mining_margin: float | None = None,
    ):
        """
        Args:
            margin (float): Triplet margin for the loss calculation.
            safe_fp_compute (bool): Compute dot-products/norms in float32 for
                numeric stability when inputs are lower precision (fp16).
            mining_strategy (str): 'hard', 'semi-hard', or 'all'.
                - 'hard': Selects the hardest negative (min d_an).
                - 'semi-hard': Selects the hardest negative *from the
                  semi-hard region* (d_ap < d_an < d_ap + mining_margin).
                - 'all': Selects *all* valid (anchor, positive, negative)
                  triplets that violate the margin (loss > 0).
            mining_margin (Optional[float]): Defines the upper bound for
                semi-hard mining. If None, defaults to `margin`.
        """
        self.margin = margin
        self.safe_fp_compute = safe_fp_compute
        self.pdist_func = lambda x, y: _euclidean_distance_matrix(
            x, y, safe_compute=self.safe_fp_compute
        )

        if mining_strategy not in ("hard", "semi-hard", "all"):
            raise ValueError(f"Unknown mining_strategy: {mining_strategy}")
        self.mining_strategy = mining_strategy

        # If no specific mining_margin is given, default it to the loss margin
        self.mining_margin = mining_margin if mining_margin is not None else self.margin

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

    def _mine_hard_or_semi_hard(
        self,
        local_dist_matrix: torch.Tensor,
        local_valid_pos_mask: torch.Tensor,
        local_valid_neg_mask: torch.Tensor,
        start_idx: int,
        local_num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function for 'hard' and 'semi-hard' mining."""

        # --- 7a) Find hardest positives (anchor-positive distance) ---
        # (Returns -inf for anchors with no valid positives)
        pos_dists = torch.where(local_valid_pos_mask, local_dist_matrix, -torch.inf)
        hard_pos_dist, hard_pos_idx = torch.max(pos_dists, dim=1)  # [local_N]

        # --- 7b) Find negatives (anchor-negative distance) ---
        if self.mining_strategy == "hard":
            # (Returns +inf for anchors with no valid negatives)
            neg_dists = torch.where(local_valid_neg_mask, local_dist_matrix, torch.inf)

        elif self.mining_strategy == "semi-hard":
            # 'Semi-hard' mining: find smallest distance d_an such that:
            #   d_ap < d_an < d_ap + self.mining_margin

            # Need d_ap broadcastable to [local_N, N]
            d_ap_broadcast = hard_pos_dist.unsqueeze(1)

            # Mask for negatives *outside* the positive (d_an > d_ap)
            mask_gt_pos = local_dist_matrix > d_ap_broadcast

            # Mask for negatives *inside* the mining margin (d_an < d_ap + mining_margin)
            mask_lt_margin = local_dist_matrix < (d_ap_broadcast + self.mining_margin)

            # Combine all masks
            semi_hard_candidate_mask = (
                local_valid_neg_mask & mask_gt_pos & mask_lt_margin
            )

            # (Returns +inf for anchors with no valid *semi-hard* negatives)
            neg_dists = torch.where(
                semi_hard_candidate_mask, local_dist_matrix, torch.inf
            )
        else:
            raise ValueError(
                f"Internal error: Unknown mining_strategy: {self.mining_strategy}"
            )

        # Find the hardest negative based on the logic above
        # (hard_neg_dist will be +inf if no valid negative was found)
        hard_neg_dist, hard_neg_idx = torch.min(neg_dists, dim=1)  # [local_N]

        # 8) Combined filter for valid, violating triplets
        # We check for margin violation: d_ap - d_an + margin > 0
        # This single check implicitly handles:
        # 1. Anchors with no valid positives (d_ap = -inf -> violation = -inf)
        # 2. Anchors with no valid negatives (d_an = +inf -> violation = -inf)
        # 3. Anchors with no semi-hard negatives (d_an = +inf -> violation = -inf)
        # 4. Valid anchors that don't violate the margin (d_ap - d_an + margin <= 0)
        triplet_violation = hard_pos_dist - hard_neg_dist + self.margin
        violating_mask_global = triplet_violation > 0  # [local_N]

        if not violating_mask_global.any():
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # 9) Select the violating triplets
        # Get local anchor indices (0...local_N-1) where violation occurs
        anchor_idx_local = torch.arange(local_num_nodes, device=device)[
            violating_mask_global
        ]

        # Convert local anchor indices to global (start_idx ... start_idx + local_N-1)
        anchors_global = (anchor_idx_local + start_idx).to(torch.long)

        # Positive and negative indices are already global (0...N-1)
        positives_global = hard_pos_idx[violating_mask_global].to(torch.long)
        negatives_global = hard_neg_idx[violating_mask_global].to(torch.long)

        return anchors_global, positives_global, negatives_global

    def _mine_all(
        self,
        local_dist_matrix: torch.Tensor,
        local_valid_pos_mask: torch.Tensor,
        local_valid_neg_mask: torch.Tensor,
        start_idx: int,
        local_num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function for 'all' (batch-all) mining."""

        all_a, all_p, all_n = [], [], []

        # Iterate over each anchor in our local slice
        for i in range(local_num_nodes):
            anchor_global_idx = start_idx + i
            dists_row = local_dist_matrix[i]  # [N]

            # Get global indices of valid positives and negatives for this anchor
            pos_mask_row = local_valid_pos_mask[i]
            neg_mask_row = local_valid_neg_mask[i]

            pos_indices_global = torch.where(pos_mask_row)[0]  # [num_pos]
            neg_indices_global = torch.where(neg_mask_row)[0]  # [num_neg]

            # If either is empty, no triplets can be formed for this anchor.
            # This implicitly handles the 'valid_anchor_mask_local' check.
            if len(pos_indices_global) == 0 or len(neg_indices_global) == 0:
                continue

            # Get the distances for these valid pairs
            # d_ap_vec: [num_pos, 1]
            # d_an_vec: [1, num_neg]
            d_ap_vec = dists_row[pos_indices_global].unsqueeze(1)
            d_an_vec = dists_row[neg_indices_global].unsqueeze(0)

            # Compute a [num_pos, num_neg] matrix of loss values
            # (d_ap - d_an + margin)
            violation_matrix = (d_ap_vec - d_an_vec + self.margin) > 0

            if not violation_matrix.any():
                continue

            # Find the (pos, neg) pairs that violate the margin
            # These indices are *local* to pos_indices_global and neg_indices_global
            violating_pos_local_idx, violating_neg_local_idx = torch.where(
                violation_matrix
            )

            # Map back to global indices
            violating_pos_global = pos_indices_global[violating_pos_local_idx]
            violating_neg_global = neg_indices_global[violating_neg_local_idx]

            # Create a tensor of anchor indices to match
            num_violations = len(violating_pos_global)
            violating_anchors_global = torch.full(
                (num_violations,),
                anchor_global_idx,
                dtype=torch.long,
                device=device,
            )

            all_a.append(violating_anchors_global)
            all_p.append(violating_pos_global)
            all_n.append(violating_neg_global)

        # Check if we found any triplets at all
        if not all_a:
            empty = torch.tensor([], dtype=torch.long, device=device)
            return empty, empty, empty

        # Concatenate lists of tensors from all anchors
        anchors_global = torch.cat(all_a)
        positives_global = torch.cat(all_p)
        negatives_global = torch.cat(all_n)

        return anchors_global, positives_global, negatives_global

    def mine(
        self, embeddings: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard, semi-hard, or all triplets for anchors in this rank's slice.

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
        rank, world_size = get_rank(), get_world_size()

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

        # Final valid masks (exclude self-comparisons)
        local_valid_pos_mask = local_pos_mask & ~local_identity_mask
        local_valid_neg_mask = local_neg_mask & ~local_identity_mask

        # 6) Dispatch to the correct mining strategy
        #    The check for anchors with valid pos/neg pairs is now handled
        #    implicitly inside the helper functions.
        if self.mining_strategy == "all":
            return self._mine_all(
                local_dist_matrix,
                local_valid_pos_mask,
                local_valid_neg_mask,
                start_idx,
                local_num_nodes,
                device,
            )
        else:
            # "hard" or "semi-hard"
            return self._mine_hard_or_semi_hard(
                local_dist_matrix,
                local_valid_pos_mask,
                local_valid_neg_mask,
                start_idx,
                local_num_nodes,
                device,
            )
