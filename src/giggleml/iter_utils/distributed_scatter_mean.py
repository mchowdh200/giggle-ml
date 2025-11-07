import torch
import torch.distributed as dist
from torch import Tensor

from giggleml.utils.autograd_aware_dist_ops import all_reduce_sum


def distributed_scatter_mean(
    local: Tensor, indices: Tensor, group: dist.ProcessGroup | None = None
) -> Tensor:
    """
    performs a scatter mean as if all ranks first did a concat.
    assumes all indices are part of a contiguous set.

    [ N, ...R ] per rank -> [ I+1, ...R ]
    where N is the major dimension of the local tensor for each rank and I is the maximum
    of all indices across all ranks.

    Crucially, this preserves the computational grad graph.
    """

    assert indices.shape == local.shape, (
        f"Indices shape ({indices.shape}) must match local tensor shape ({local.shape})"
    )

    # ensure indices are long for scatter_add_
    indices = indices.long()

    # get max index
    world_size = dist.get_world_size(group=group)
    max_indices = torch.zeros(world_size, dtype=indices.dtype, device=indices.device)
    dist.all_gather_into_tensor(max_indices, indices.max().unsqueeze(0), group=group)
    max_idx = int(max_indices.max().item())

    # create output tensors
    sums = torch.zeros(
        max_idx + 1, *local.shape[1:], dtype=local.dtype, device=local.device
    )
    counts = torch.zeros(
        max_idx + 1, *local.shape[1:], dtype=local.dtype, device=local.device
    )

    # scatter sum
    sums.scatter_add_(0, indices, local)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=local.dtype))

    # all gather
    all_reduce_sum(sums, group=group)
    all_reduce_sum(counts, group=group)

    # mean
    return sums / counts
