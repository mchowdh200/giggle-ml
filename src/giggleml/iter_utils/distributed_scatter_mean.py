import torch
import torch.distributed as dist
from torch import Tensor

from giggleml.utils.autograd_aware_dist_ops import all_reduce_sum


def distributed_scatter_mean(
    local: Tensor, indices: Tensor, group: dist.ProcessGroup | None = None
) -> Tensor:
    """
    Performs a distributed scatter_mean operation on a per-row basis.
    Assumes all tensors were virtually concatenated along dimension 0.

    [ N, ...R ] per rank -> [ I+1, ...R ]
    where N is the number of rows (dim 0) of the local tensor for each rank,
    and I is the maximum of all indices across all ranks.

    Crucially, this preserves the computational grad graph.
    """

    # --- 1. Shape Assertion ---
    assert indices.ndim == 1, (
        f"Indices must be a 1D tensor, but got {indices.ndim} dimensions."
    )
    assert indices.shape[0] == local.shape[0], (
        f"Indices shape ({indices.shape}) must match local tensor's dim 0 ({local.shape})"
    )

    # Ensure indices are long for scatter_add_
    indices_long = indices.long()
    local_device = local.device
    local_dtype = local.dtype

    # --- 2. Get Global Max Index ---
    # Find local max
    local_max_idx = indices_long.max()

    # Create a tensor to hold the global max
    # We clone to ensure a clean tensor for the in-place all_reduce
    global_max_idx_tensor = local_max_idx.clone()

    # Reduce across all ranks to find the true maximum
    dist.all_reduce(global_max_idx_tensor, op=dist.ReduceOp.MAX, group=group)

    max_idx = int(global_max_idx_tensor.item())
    output_size = max_idx + 1

    # --- 3. Create Output Tensors ---
    sums = torch.zeros(
        output_size, *local.shape[1:], dtype=local_dtype, device=local_device
    )

    # We can use a smaller dtype if we don't expect > 2.1B items per index
    counts = torch.zeros(output_size, dtype=torch.int64, device=local_device)

    # --- 4. Local Scatter-Sum & Scatter-Count ---

    # For sums, we need to expand indices to match the shape of `local`
    # (N,) -> (N, 1, 1, ...)
    indices_expanded = indices_long.view(-1, *([1] * (local.ndim - 1)))
    # (N, 1, 1, ...) -> (N, *R)
    indices_expanded = indices_expanded.expand_as(local)

    sums.scatter_add_(0, indices_expanded, local)

    # For counts (1D), we just scatter 1s using the 1D indices
    ones = torch.ones_like(indices_long, dtype=torch.int64)
    counts.scatter_add_(0, indices_long, ones)

    # --- 5. All-Reduce (using autograd-aware op) ---
    # `sums` needs gradients, `counts` does not
    all_reduce_sum(sums, group=group)

    # `counts` does not require gradients, so a standard all_reduce is fine
    # and likely more efficient if `all_reduce_sum` has autograd overhead.
    # If `all_reduce_sum` handles this, using it is also fine.
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)

    # --- 6. Compute Mean with Safe Division ---
    result = torch.zeros_like(sums)

    # Find indices where count is non-zero
    non_zero_mask = counts > 0

    # Get the broadcastable counts for division
    # (I+1) -> (I+1, 1, 1, ...)
    counts_broadcastable = counts.view(-1, *([1] * (local.ndim - 1)))
    # Cast to float for division
    counts_broadcastable = counts_broadcastable.to(local_dtype)

    # Apply mask to all relevant tensors
    non_zero_indices = non_zero_mask.nonzero().squeeze(-1)

    # Only compute division where count is not zero
    # We index to avoid 0/0 -> NaN
    result[non_zero_indices] = (
        sums[non_zero_indices] / counts_broadcastable[non_zero_indices]
    )

    return result
