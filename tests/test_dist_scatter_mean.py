import torch

from giggleml.iter_utils.distributed_scatter_mean import (
    distributed_scatter_mean,
    distributed_scatter_mean_iter,
)
from giggleml.utils.parallel import Parallel
from giggleml.utils.torch_utils import get_rank, get_world_size


def test_all_ranks_empty():
    Parallel(world_size=3).run(_test_all_ranks_empty)


def _test_all_ranks_empty():
    """
    Tests the case where no rank provides any data.
    The result should be an empty dictionary.
    """
    indices = []
    data = []

    result = distributed_scatter_mean(indices, data)

    assert result == {}, "Expected an empty dict when no data is provided"
    print("test_all_ranks_empty: PASSED")


def test_one_rank_empty():
    Parallel(world_size=3).run(_test_one_rank_empty)


def _test_one_rank_empty():
    """
    Tests the case where at least one rank has data, but rank 0 does not.
    This validates the prototype tensor discovery logic.
    (Requires world_size > 1)
    """
    rank = get_rank()
    world_size = get_world_size()

    if world_size == 1:
        print("test_one_rank_empty: SKIPPED (requires world_size > 1)")
        return

    if rank == 0:
        indices = []
        data = []
    else:
        # All other ranks contribute to set 5
        indices = [5, 5]
        data = [torch.tensor([2.0]), torch.tensor([4.0])]  # local mean is 3.0

    result = distributed_scatter_mean(indices, data)

    # Each of the (world_size - 1) ranks contributes a sum of 6.0 and count of 2.
    # Global sum = 6.0 * (world_size - 1)
    # Global count = 2 * (world_size - 1)
    # Global mean = (6.0 * (ws - 1)) / (2 * (ws - 1)) = 3.0
    expected_mean = torch.tensor([3.0])

    assert 5 in result, "Set 5 should be present in the result"
    assert torch.allclose(result[5], expected_mean), (
        f"Expected mean {expected_mean}, got {result[5]}"
    )
    print("test_one_rank_empty: PASSED")


def test_basic_overlap():
    Parallel(world_size=3).run(_test_basic_overlap)


def _test_basic_overlap():
    """
    Tests the primary use case where ranks have overlapping and unique sets.
    - Set 100 is shared by all ranks.
    - Set `rank` is unique to each rank.
    """
    rank = get_rank()
    world_size = get_world_size()

    # Each rank contributes [1., 2.] to set 100
    # and [rank, rank] to its own unique set.
    indices = [100, rank, 100]
    data = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([float(rank), float(rank)]),
        torch.tensor([1.0, 2.0]),  # Add a second item for set 100
    ]

    result = distributed_scatter_mean(indices, data)

    # --- Check expectations ---

    # 1. Check the shared set (100)
    # Each rank's local sum for 100 is [2.0, 4.0], count is 2.
    # Global sum = [2.0, 4.0] * world_size
    # Global count = 2 * world_size
    # Global mean = ([2.0, 4.0] * ws) / (2 * ws) = [1.0, 2.0]
    expected_mean_100 = torch.tensor([1.0, 2.0])
    assert 100 in result
    assert torch.allclose(result[100], expected_mean_100), (
        f"Set 100 failed: {result[100]}"
    )

    # 2. Check the unique set for this rank
    # Global sum = [rank, rank] * 1 (from one rank)
    # Global count = 1
    # Global mean = [rank, rank]
    # Note: data was [float(rank), float(rank)] and there was one such item.
    # The local sum is [rank, rank], count is 1.
    # Global sum is [rank, rank], count is 1.
    # Mean is [rank, rank]. This looks wrong in the original test logic.
    # Let's re-check the data for test_basic_overlap:
    # data = [
    #     torch.tensor([1.0, 2.0]), # -> set 100
    #     torch.tensor([float(rank), float(rank)]), # -> set rank
    #     torch.tensor([1.0, 2.0]), # -> set 100
    # ]
    # Local sum for set `rank` is [rank, rank], count is 1.
    # Global sum for set `rank` (from rank `rank`) is [rank, rank], count is 1.
    # Global mean is [rank, rank] / 1 = [rank, rank].
    # Ah, the original logic was correct.
    expected_mean_rank = torch.tensor([float(rank), float(rank)])
    assert rank in result
    assert torch.allclose(result[rank], expected_mean_rank), (
        f"Set {rank} failed: {result[rank]}"
    )

    # 3. Check total number of keys
    expected_num_keys = world_size + 1  # (sets 0..ws-1) + set 100
    assert len(result) == expected_num_keys, (
        f"Expected {expected_num_keys} keys, got {len(result)}"
    )

    print("test_basic_overlap: PASSED")


def test_iterator_wrapper():
    Parallel(world_size=3).run(_test_iterator_wrapper)


def _test_iterator_wrapper():
    """
    Tests that the iterator wrapper provides the same data, sorted by key.
    """
    rank = get_rank()

    # Simpler data than test_basic_overlap to avoid confusion
    indices = [100, rank]
    data = [torch.tensor([1.0, 2.0]), torch.tensor([float(rank), float(rank)])]

    # Get the dictionary result first
    dict_result = distributed_scatter_mean(indices, data)
    expected_list = sorted(dict_result.items())

    # Get the iterator result
    iter_result = list(distributed_scatter_mean_iter(indices, data))

    assert len(iter_result) == len(expected_list), (
        "Iterator and dict have different lengths"
    )

    for (iter_key, iter_val), (exp_key, exp_val) in zip(iter_result, expected_list):
        assert iter_key == exp_key, "Iterator keys are not sorted or do not match"
        assert torch.allclose(iter_val, exp_val), "Iterator values do not match"

    print("test_iterator_wrapper: PASSED")


def test_autograd_differentiability():
    Parallel(world_size=3).run(_test_autograd_differentiability)


def _test_autograd_differentiability():
    """
    Tests that gradients can flow back through the scatter-mean operation.
    """
    rank = get_rank()
    world_size = get_world_size()

    # Input tensors that require gradients
    t1 = torch.tensor([1.0, 2.0], requires_grad=True)
    t2 = torch.tensor([float(rank), 10.0], requires_grad=True)

    indices = [100, rank]  # Set 100 is shared, set `rank` is unique
    data = [t1, t2]

    # Run the function
    result = distributed_scatter_mean(indices, data)

    # All ranks sum up their results. This is a common pattern.
    # The sum() must be identical across all ranks.

    # Initialize total_sum as a 0-dim tensor, not a Python float.
    # This preserves the computation graph.
    total_sum = torch.tensor(0.0)
    for key in sorted(result.keys()):  # Sort for deterministic sum
        # This is now tensor addition
        total_sum = total_sum + result[key].sum()

    # Perform backward pass
    try:
        total_sum.backward()
    except RuntimeError as e:
        assert False, f"Backward pass failed: {e}"

    # --- Check gradients ---

    # 1. Grad for t1 (shared set 100)
    # result[100] = (t1_rank0 + t1_rank1 + ...) / world_size
    # total_sum = result[100].sum() + ...
    # d(total_sum) / d(t1_rankN) = d(result[100].sum()) / d(t1_rankN)
    # = d( (t1_rank0.sum() + ...) / world_size ) / d(t1_rankN)
    # = (1 / world_size) * d(t1_rankN.sum()) / d(t1_rankN)
    # = (1 / world_size) * 1.0 (for each element)
    assert t1.grad is not None, "t1 has no gradient"
    expected_t1_grad = torch.ones_like(t1) / world_size
    assert torch.allclose(t1.grad, expected_t1_grad), f"t1 grad mismatch: {t1.grad}"

    # 2. Grad for t2 (unique set `rank`)
    # result[rank] = (t2_rank) / 1
    # total_sum = ... + result[rank].sum() + ...
    # d(total_sum) / d(t2_rank) = d(result[rank].sum()) / d(t2_rank)
    # = d( t2_rank.sum() / 1 ) / d(t2_rank) = 1.0
    assert t2.grad is not None, "t2 has no gradient"
    expected_t2_grad = torch.ones_like(t2)
    assert torch.allclose(t2.grad, expected_t2_grad), f"t2 grad mismatch: {t2.grad}"

    print("test_autograd_differentiability: PASSED")
