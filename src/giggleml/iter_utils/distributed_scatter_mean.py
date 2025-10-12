from collections import defaultdict
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

from giggleml.utils.torch_utils import get_world_size, is_distributed


def distributed_scatter_mean(
    set_indices: Iterable[int],
    rank_data: Iterable[torch.Tensor],
) -> dict[int, torch.Tensor]:
    """
    Computes distributed set-wise means over scattered data.

    This function orchestrates a three-phase process:
    1.  **Local Aggregation**: Each rank iterates through its slice of data (tensors),
        summing values and counting items for each set ID.
    2.  **Global Gathering**: The local sums and counts from all ranks are
        collected using a distributed `all_gather` operation.
    3.  **Final Computation**: The global sums are divided by the global counts
        to produce the final mean for each set.

    All ranks receive the identical, complete set of means.

    Args:
        set_indices: An iterable that yields a flat sequence of set IDs,
            corresponding to each tensor in `rank_data`.
        rank_data: An iterable of tensors for the current rank.

    Returns:
        A dictionary mapping each set ID to its computed mean tensor. The
        result is identical across all ranks.
    """
    # Phase 1: Perform scatter-add on local data.
    local_sums, local_counts = _local_aggregate(set_indices, rank_data)

    # Phase 2: Gather results from all ranks if in a distributed environment.
    if is_distributed():
        global_sums, global_counts = _all_gather_and_reduce(local_sums, local_counts)
    else:
        global_sums, global_counts = local_sums, local_counts

    # Phase 3: Compute the final averages from global aggregates.
    return _compute_averages(global_sums, global_counts)


def scatter_mean_iter(
    set_indices: Iterable[int],
    rank_data: Iterable[torch.Tensor],
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Computes all set means and yields them as an iterator.

    Note: This is not a true streaming computation, as all data must be
    gathered before any means can be calculated. It provides an iterator
    interface to the final results.

    Args:
        set_indices: An iterable that yields a flat sequence of set IDs,
            corresponding to each tensor in `rank_data`.
        rank_data: An iterable of tensors for the current rank.

    Yields:
        Tuples of (set_id, mean_tensor), sorted by set_id for determinism.
    """
    means = scatter_mean(set_indices, rank_data)
    # Yield results sorted by key for deterministic ordering.
    yield from sorted(means.items())


def _local_aggregate(
    set_indices: Iterable[int],
    rank_data: Iterable[torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """Aggregates sums and counts for data items on the local rank."""
    set_sums: dict[int, torch.Tensor] = {}
    # Use defaultdict for cleaner accumulation of counts.
    set_counts: dict[int, int] = defaultdict(int)

    for set_idx, tensor in zip(set_indices, rank_data):
        # Move tensor to CPU for aggregation to avoid cross-device issues
        # and centralize data.
        tensor = tensor.cpu()

        # Accumulate sum and count for the corresponding set.
        if set_idx in set_sums:
            set_sums[set_idx].add_(tensor)  # In-place add for efficiency
        else:
            set_sums[set_idx] = tensor.clone()
        set_counts[set_idx] += 1

    return set_sums, set_counts


def _all_gather_and_reduce(
    local_sums: dict[int, torch.Tensor], local_counts: dict[int, int]
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """
    Gathers local aggregates from all ranks and reduces them into global ones.

    Note: `dist.all_gather_object` can be a performance bottleneck as it
    relies on pickling. For extreme performance needs, consider a fully
    tensor-based communication strategy using `dist.all_gather`.
    """
    world_size = get_world_size()
    gathered_sums: list[dict[int, torch.Tensor]] = [{} for _ in range(world_size)]
    gathered_counts: list[dict[int, int]] = [{} for _ in range(world_size)]

    dist.all_gather_object(gathered_sums, local_sums)
    dist.all_gather_object(gathered_counts, local_counts)

    # Reduce the gathered lists into single global dictionaries.
    global_sums: dict[int, torch.Tensor] = {}
    global_counts: dict[int, int] = defaultdict(int)

    for rank_sums, rank_counts in zip(gathered_sums, gathered_counts):
        for set_idx, count in rank_counts.items():
            global_counts[set_idx] += count
            tensor_sum = rank_sums[set_idx]
            if set_idx in global_sums:
                global_sums[set_idx].add_(tensor_sum)
            else:
                global_sums[set_idx] = tensor_sum.clone()

    return global_sums, global_counts


def _compute_averages(
    global_sums: dict[int, torch.Tensor], global_counts: dict[int, int]
) -> dict[int, torch.Tensor]:
    """Computes the final mean from global sums and counts."""
    # A dictionary comprehension is a concise way to perform the final division.
    return {
        set_idx: global_sums[set_idx] / count
        for set_idx, count in global_counts.items()
        if count > 0
    }
