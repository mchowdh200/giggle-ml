import torch
import torch.distributed as dist
from torch.testing import assert_close

from giggleml.utils.all_gather_cat import (
    all_gather_cat,  # Better for floats than torch.equal
)
from giggleml.utils.parallel import Parallel


def get_rank_and_world_size(group=None):
    """Helper to get rank/world_size for default or specific group."""
    return dist.get_rank(group), dist.get_world_size(group)


def test_forward_pass_even_sizes():
    if not torch.cuda.is_available():
        print("skipping test case because we're not on cuda")
    Parallel(3).run(_test_forward_pass_even_sizes)


def _test_forward_pass_even_sizes():
    """
    Tests forward pass with all tensors having the same size.
    """
    rank, world_size = get_rank_and_world_size()
    device = torch.device(f"cuda:{rank}")

    # Each rank creates a tensor with its rank value
    # Size: (2, 4)
    local_tensor = torch.full(
        (2, 4), fill_value=rank, dtype=torch.float32, device=device
    )

    # Run the operation
    global_tensor = all_gather_cat(local_tensor)

    # --- Check Shape ---
    assert global_tensor.shape == (world_size * 2, 4)

    # --- Check Content ---
    expected_chunks = []
    for i in range(world_size):
        expected_chunks.append(
            torch.full((2, 4), fill_value=i, dtype=torch.float32, device=device)
        )
    expected_global = torch.cat(expected_chunks, dim=0)

    assert_close(global_tensor, expected_global)


def test_forward_and_backward_pass_uneven_sizes():
    if not torch.cuda.is_available():
        print("skipping test case because we're not on cuda")
    Parallel(3).run(_test_forward_and_backward_pass_uneven_sizes)


def _test_forward_and_backward_pass_uneven_sizes():
    """
    Tests both forward and backward pass with *uneven* tensor sizes.

    This is the most important test. It simulates a "per-rank" loss
    by having each rank only sum its own contribution to the global
    tensor. The backward pass should correctly sum all partial
    gradients and slice the correct part for each rank.
    """
    rank, world_size = get_rank_and_world_size()
    device = torch.device(f"cuda:{rank}")

    # --- Setup ---
    # Rank 'i' creates a tensor of size (i + 1, 4)
    local_batch_size = rank + 1
    local_tensor = torch.randn(
        (local_batch_size, 4), dtype=torch.float32, device=device, requires_grad=True
    )

    # We need the sizes of all tensors to check the forward pass
    # and to compute the per-rank loss
    all_sizes = list(range(1, world_size + 1))

    # --- Forward Pass ---
    global_tensor = all_gather_cat(local_tensor)

    # Check shape
    expected_global_size = sum(all_sizes)
    assert global_tensor.shape == (expected_global_size, 4)

    # --- Backward Pass (Simulating Per-Rank Loss) ---

    # Calculate this rank's slice from the global tensor
    start_idx = sum(all_sizes[:rank])
    end_idx = start_idx + all_sizes[rank]  # or start_idx + local_batch_size

    # Compute loss ONLY on this rank's part of the global tensor
    # This simulates "mining for local anchors"
    loss = global_tensor[start_idx:end_idx].sum()

    # Backpropagate
    loss.backward()

    # --- Check Gradient ---
    #
    # How this works:
    # 1. Rank 'i' computes a loss, creating a gradient that is
    #    1.0 for its slice and 0.0 everywhere else.
    # 2. The custom backward() calls all_reduce(op=SUM).
    # 3. The final gradient tensor (before slicing) becomes all 1s,
    #    because each slice's gradient was contributed by exactly one rank.
    # 4. The backward() then slices this all-1s tensor.
    #
    # Therefore, the local gradient should be all 1s.
    expected_grad = torch.ones_like(local_tensor)
    assert_close(local_tensor.grad, expected_grad)


def test_process_group_support():
    if not torch.cuda.is_available():
        print("skipping test case because we're not on cuda")
    Parallel(3).run(_test_process_group_support)


def _test_process_group_support():
    """
    Tests that the function works correctly with a subgroup.
    (This test will only run if world_size is >= 2)
    """
    rank, world_size = get_rank_and_world_size()
    if world_size < 2:
        print("Skipping process group test, world_size < 2")
        return

    device = torch.device(f"cuda:{rank}")

    # Create a subgroup of the first two ranks
    group_ranks = [0, 1]
    group = dist.new_group(ranks=group_ranks)

    global_tensor = None

    if rank in group_ranks:
        # Only ranks in the group participate
        group_rank, group_world_size = get_rank_and_world_size(group)
        assert group_world_size == 2

        # Size (group_rank + 1, 4) -> Rank 0: (1, 4), Rank 1: (2, 4)
        local_tensor = torch.full(
            (group_rank + 1, 4),
            fill_value=float(group_rank),
            dtype=torch.float32,
            device=device,
        )

        global_tensor = all_gather_cat(local_tensor, group=group)

        # --- Check Forward Pass (within group) ---
        # Total size = 1 (from rank 0) + 2 (from rank 1) = 3
        assert global_tensor.shape == (3, 4)

        # Check content
        chunk_0 = torch.full((1, 4), 0.0, device=device)
        chunk_1 = torch.full((2, 4), 1.0, device=device)
        expected_global = torch.cat([chunk_0, chunk_1], dim=0)

        assert_close(global_tensor, expected_global)

    else:
        # Ranks not in the group should not have run the op
        assert global_tensor is None
