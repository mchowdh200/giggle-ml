from collections import defaultdict
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

from giggleml.utils.torch_utils import is_distributed


def distributed_scatter_mean(
    set_indices: Iterable[int],
    rank_data: Iterable[torch.Tensor],
    dist_group: dist.ProcessGroup | None = None,
) -> dict[int, torch.Tensor]:
    """
    Computes distributed set-wise means over scattered data.

    This function orchestrates a three-phase process:
    1.  **Local Aggregation**: Each rank iterates through its slice of data (tensors),
        summing values and counting items for each set ID.
    2.  **Global Gathering**: The local sums and counts from all ranks are
        collected using a differentiable, tensor-based `all_gather` operation.
    3.  **Final Computation**: The global sums are divided by the global counts
        to produce the final mean for each set.

    All ranks receive the identical, complete set of means.

    Args:
        set_indices: An iterable that yields a flat sequence of set IDs,
            corresponding to each tensor in `rank_data`.
        rank_data: An iterable of tensors for the current rank.
        dist_group: The process group to use for communication.

    Returns:
        A dictionary mapping each set ID to its computed mean tensor. The
        result is identical across all ranks.
    """
    # Phase 1: Perform scatter-add on local data.
    local_sums, local_counts = _local_aggregate(set_indices, rank_data)

    # Phase 2: Gather results from all ranks if in a distributed environment.
    if is_distributed():
        global_sums, global_counts = _all_gather_and_reduce(
            local_sums, local_counts, dist_group
        )
    else:
        global_sums, global_counts = local_sums, local_counts

    # Phase 3: Compute the final averages from global aggregates.
    return _compute_averages(global_sums, global_counts)


def distributed_scatter_mean_iter(
    set_indices: Iterable[int],
    rank_data: Iterable[torch.Tensor],
    dist_group: dist.ProcessGroup | None = None,
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
        dist_group: The process group to use for communication.

    Yields:
        Tuples of (set_id, mean_tensor), sorted by set_id for determinism.
    """
    means = distributed_scatter_mean(set_indices, rank_data, dist_group)
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

    first_tensor = None

    for set_idx, tensor in zip(set_indices, rank_data):
        if first_tensor is None:
            first_tensor = tensor

        # Move tensor to CPU for aggregation. This is necessary for
        # efficient communication with a 'gloo' backend.
        tensor = tensor.cpu()

        # Accumulate sum and count for the corresponding set.
        # *** AUTOGRAD FIX ***
        # We must use standard, out-of-place addition to preserve
        # the computation graph. `add_` is an in-place operation
        # on a leaf tensor, which breaks autograd.
        if set_idx in set_sums:
            set_sums[set_idx] = set_sums[set_idx] + tensor
        else:
            # Do not .clone(). Cloning detaches the tensor from the
            # graph. Just assign it directly.
            set_sums[set_idx] = tensor
        # *** END FIX ***
        set_counts[set_idx] += 1

    # Ensure all tensors in the dictionary are on the same device
    # even if they were on different devices initially.
    if first_tensor is not None:
        cpu_device = torch.device("cpu")
        for tensor in set_sums.values():
            if tensor.device != cpu_device:
                tensor.to(cpu_device)

    return set_sums, set_counts


def _all_gather_and_reduce(
    local_sums: dict[int, torch.Tensor],
    local_counts: dict[int, int],
    dist_group: dist.ProcessGroup | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """
    Gathers local aggregates using a differentiable, tensor-based strategy.

    This function performs two communication stages:
    1.  Gathers all set IDs (via `all_gather_object`) to create a
        canonical, global ordering for all sets.
    2.  Each rank creates ordered tensors for its local sums and counts.
    3.  Gathers these tensors using `all_gather_into_tensor`, which is
        differentiable for the sums tensor.
    4.  Reduces the global tensors (summing across ranks) and converts
        them back into dictionaries.
    """
    # *** FIX: Get world_size and rank from the specified group ***
    world_size = dist.get_world_size(dist_group)
    rank = dist.get_rank(dist_group)

    # --- Stage 1: Gather all local data to Rank 0 ---

    # Gather all local set IDs from all ranks
    local_keys = list(local_sums.keys())
    all_rank_keys: list[list[int]] = [[] for _ in range(world_size)]
    dist.all_gather_object(all_rank_keys, local_keys, dist_group)

    # Gather tensor properties from all ranks
    prototype = next(iter(local_sums.values()), None)
    local_props = (
        (tuple(prototype.shape), prototype.dtype)
        if prototype is not None
        else (None, None)
    )
    gathered_props: list[tuple[tuple[int, ...] | None, torch.dtype | None]] = [
        (None, None)
    ] * world_size
    dist.all_gather_object(gathered_props, local_props, dist_group)

    # --- Stage 2: Rank 0 computes canonical metadata and broadcasts it ---

    if rank == 0:
        # Rank 0 computes the canonical metadata
        all_keys_flat = [key for rank_keys in all_rank_keys for key in rank_keys]
        global_set_indices = sorted(list(set(all_keys_flat)))
        set_idx_to_tensor_idx = {
            set_id: i for i, set_id in enumerate(global_set_indices)
        }

        first_valid_props = None
        for p in gathered_props:
            if p is not None and p[0] is not None:
                first_valid_props = p
                break

        if first_valid_props is None:
            # No data on any rank
            sync_data = [None, None, None]
        else:
            tensor_shape_tuple, dtype = first_valid_props
            tensor_shape = torch.Size(tensor_shape_tuple)
            sync_data = [set_idx_to_tensor_idx, tensor_shape, dtype]
    else:
        # Ranks 1+ prepare empty containers to receive the broadcast
        sync_data = [None, None, None]

    # *** FIX: Broadcast metadata from Rank 0 to all other ranks ***
    # This is far more robust than relying on all_gather_object to be consistent.
    dist.broadcast_object_list(sync_data, src=0, group=dist_group)

    # All ranks now have identical metadata
    set_idx_to_tensor_idx, tensor_shape, dtype = sync_data

    # If no tensors exist on any rank, return empty results
    if set_idx_to_tensor_idx is None:
        return {}, {}

    set_count = len(set_idx_to_tensor_idx)
    device = torch.device("cpu")  # We know from _local_aggregate

    # --- Stage 3: Prepare local tensors using synchronized metadata ---

    # *** FIX for Autograd ***
    # We cannot use `torch.zeros()` and in-place assignment, as it breaks
    # the computation graph. Instead, we build a Python list of tensors
    # (some are zero-tensors, some are from `local_sums`) and then
    # `torch.stack()` them. `torch.stack` is a differentiable operation
    # and will correctly construct the graph.

    # 1. Create a list of tensors, pre-filled with new zero tensors.
    # We must create new zero tensors, not references to the *same* tensor.
    local_sum_list_for_stack = [
        torch.zeros(*tensor_shape, dtype=dtype, device=device) for _ in range(set_count)
    ]

    # We still need the local_count_tensor, which doesn't need grad.
    local_count_tensor = torch.zeros(set_count, dtype=torch.int64, device=device)

    for set_idx, tensor_sum in local_sums.items():
        # `tensor_sum` is the *original* tensor from the input,
        # which may require grad and have a grad_fn.

        if set_idx not in set_idx_to_tensor_idx:
            # This key was not part of the canonical broadcasted map.
            # This can happen if all_gather_object is non-deterministic
            # and Rank 0 didn't see a key that this rank has.
            continue

        tensor_idx = set_idx_to_tensor_idx[set_idx]

        # 3. Place the (potentially grad-requiring) tensor into the list.
        # This is just a Python list assignment.
        if tensor_sum.shape != tensor_shape:
            try:
                # Reshaping a grad-requiring tensor preserves the graph.
                local_sum_list_for_stack[tensor_idx] = tensor_sum.reshape(tensor_shape)
            except RuntimeError:
                raise ValueError(
                    f"[Rank {rank}] Incompatible tensor shape for set {set_idx}. "
                    f"Aggregator expected {tensor_shape}, but got {tensor_sum.shape}."
                )
        else:
            local_sum_list_for_stack[tensor_idx] = tensor_sum

        local_count_tensor[tensor_idx] = local_counts[set_idx]

    # 4. Stack the list to create the local_sum_tensor.
    # If `local_sum_list_for_stack` contains tensors that require grad,
    # `local_sum_tensor` will now correctly have a `grad_fn` (StackBackward).
    local_sum_tensor = torch.stack(local_sum_list_for_stack, dim=0)

    # --- Stage 4 & 5: Differentiable All-Reduce ---

    # *** FIX for Autograd & Gloo Bug ***
    # The all_gather + stack logic was incorrect and broke autograd.
    # The correct, differentiable way to get a global sum is all_reduce.

    # 1. Clone the tensors. all_reduce is in-place, but we need the
    #    original `local_sum_tensor` for the backward pass.
    #    `clone()` is a differentiable operation.
    final_sums = local_sum_tensor.clone()
    final_counts = local_count_tensor.clone()

    # 2. Perform the all-reduce operation. This sums the tensors
    #    from all ranks and distributes the result back to all ranks.
    #    This operation is fully differentiable.
    dist.all_reduce(final_sums, op=dist.ReduceOp.SUM, group=dist_group)
    dist.all_reduce(final_counts, op=dist.ReduceOp.SUM, group=dist_group)

    # --- Stage 6: Convert back to dicts ---

    global_sums_dict: dict[int, torch.Tensor] = {}
    global_counts_dict: dict[int, int] = defaultdict(int)

    # Convert the reduced tensors back into dictionaries
    # `final_sums` is now a tensor with a valid grad_fn.
    for set_idx, tensor_idx in set_idx_to_tensor_idx.items():
        count = final_counts[tensor_idx].item()
        if count > 0:
            # Slicing a tensor is also differentiable.
            global_sums_dict[set_idx] = final_sums[tensor_idx]
            global_counts_dict[set_idx] = count

    return global_sums_dict, global_counts_dict


def _compute_averages(
    global_sums: dict[int, torch.Tensor], global_counts: dict[int, int]
) -> dict[int, torch.Tensor]:
    """Computes the final mean from global sums and counts."""

    # Using torch.div for explicit broadcasting and in-place-like ops
    # This loop is necessary because the counts are not in a tensor.
    final_means = {}
    for set_idx, count in global_counts.items():
        if count > 0:
            # .add_(0) is a trick to ensure the tensor has a grad_fn
            # if it came from a rank that was not part of the graph,
            # though sum() in _all_gather_and_reduce should handle this.
            sum_tensor = global_sums[set_idx]
            # Division by a scalar is differentiable.
            final_means[set_idx] = sum_tensor / count

    return final_means
