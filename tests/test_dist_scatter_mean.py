import torch
import torch.distributed as dist

from giggleml.iter_utils.distributed_scatter_mean import distributed_scatter_mean
from giggleml.utils.parallel import Parallel


def test_works_with_1d_feature():  # Removed rank, world_size args
    Parallel(2).run(_test_works_with_1d_feature)


def _test_works_with_1d_feature():  # Removed rank, world_size args
    """
    Tests the function with tensors of shape [N, 1], which should work.
    Assumes world_size = 2.
    """
    # Assume environment is already set up
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This test is designed to be run with world_size=2"

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Each rank has different data
    if rank == 0:
        # data: [10, 20], indices: [0, 1]
        local_data = torch.tensor([[10.0], [20.0]], dtype=torch.float32).to(device)
        indices = torch.tensor([[0], [1]], dtype=torch.long).to(device)
    else:  # rank == 1
        # data: [30, 40], indices: [0, 0]
        local_data = torch.tensor([[30.0], [40.0]], dtype=torch.float32).to(device)
        indices = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    # Calculate expected result manually
    # Index 0: 10.0 (r0) + 30.0 (r1) + 40.0 (r1) = 80.0. Count = 3. Mean = 80/3
    # Index 1: 20.0 (r0). Count = 1. Mean = 20/1
    expected_mean = torch.tensor([[80.0 / 3.0], [20.0]], dtype=torch.float32).to(device)

    # Run the function
    result = distributed_scatter_mean(local_data, indices)

    # Check result
    assert torch.allclose(result, expected_mean), (
        f"Expected {expected_mean}, but got {result}"
    )

    # Removed _cleanup_distributed()


def test_works_with_2d_feature():  # Renamed from test_fails_with_2d_feature
    Parallel(2).run(_test_works_with_2d_feature)


def _test_works_with_2d_feature():  # Renamed from test_fails_with_2d_feature
    """
    Tests the function with tensors of shape [N, 2], which should now work
    with the improved function.
    Assumes world_size = 2.
    """
    # Assume environment is already set up
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This test is designed to be run with world_size=2"

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Each rank has different data
    if rank == 0:
        local_data = torch.tensor([[10.0, 1.0], [20.0, 2.0]], dtype=torch.float32).to(
            device
        )
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.long).to(device)
    else:  # rank == 1
        local_data = torch.tensor([[30.0, 3.0], [40.0, 4.0]], dtype=torch.float32).to(
            device
        )
        indices = torch.tensor([[0, 0], [0, 0]], dtype=torch.long).to(device)

    # This function call is no longer expected to fail
    result = distributed_scatter_mean(local_data, indices)

    # --- Manual Calculation ---
    # total_sums (from all-reduce):
    # sums[0,0] = 10.0 (r0) + 30.0 (r1) + 40.0 (r1) = 80.0
    # sums[0,1] = 1.0 (r0) + 3.0 (r1) + 4.0 (r1) = 8.0
    # sums[1,0] = 20.0 (r0)
    # sums[1,1] = 2.0 (r0)
    # total_sums = [[80.0, 8.0], [20.0, 2.0]]
    #
    # total_counts (from all-reduce):
    # counts[0,0] = 1 (r0) + 1 (r1) + 1 (r1) = 3
    # counts[0,1] = 1 (r0) + 1 (r1) + 1 (r1) = 3
    # counts[1,0] = 1 (r0)
    # counts[1,1] = 1 (r0)
    # total_counts = [[3.0, 3.0], [1.0, 1.0]]
    #
    # mean = total_sums / total_counts
    expected_mean = torch.tensor(
        [[80.0 / 3.0, 8.0 / 3.0], [20.0 / 1.0, 2.0 / 1.0]], dtype=torch.float32
    ).to(device)

    # Check result
    assert torch.allclose(result, expected_mean), (
        f"Expected {expected_mean}, but got {result}"
    )

    # Removed _cleanup_distributed()


def test_gradient_flow():  # Removed rank, world_size args
    Parallel(2).run(_test_gradient_flow)


def _test_gradient_flow():  # Removed rank, world_size args
    """
    Tests that gradients can flow backward through the function.
    Uses the 1D feature case that is known to work.
    Assumes world_size = 2.
    """
    # Assume environment is already set up
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This test is designed to be run with world_size=2"

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Each rank has different data
    if rank == 0:
        local_data = torch.tensor(
            [[10.0], [20.0]], dtype=torch.float32, requires_grad=True
        ).to(device)
        indices = torch.tensor([[0], [1]], dtype=torch.long).to(device)
    else:  # rank == 1
        local_data = torch.tensor(
            [[30.0], [40.0]], dtype=torch.float32, requires_grad=True
        ).to(device)
        indices = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    # Run the function
    result = distributed_scatter_mean(local_data, indices)

    # Backpropagate a simple sum
    result.sum().backward()

    # Check if gradients are computed
    assert local_data.grad is not None

    # Manual gradient calculation:
    # result[0] = (local_0_r0 + local_0_r1 + local_1_r1) / 3
    # result[1] = (local_1_r0) / 1
    # loss = result[0] + result[1]
    #
    # Grads for Rank 0:
    # d_loss / d_local_0_r0 = 1/3
    # d_loss / d_local_1_r0 = 1
    # Expected grad_r0 = [[1/3], [1.0]]
    #
    # Grads for Rank 1:
    # d_loss / d_local_0_r1 = 1/3
    # d_loss / d_local_1_r1 = 1/3
    # Expected grad_r1 = [[1/3], [1/3]]

    if rank == 0:
        expected_grad = torch.tensor([[1.0 / 3.0], [1.0]], device=device)
    else:
        expected_grad = torch.tensor([[1.0 / 3.0], [1.0 / 3.0]], device=device)

    assert torch.allclose(local_data.grad, expected_grad)


def test_works_with_1d_tensor():
    Parallel(2).run(_test_works_with_1d_tensor)


def _test_works_with_1d_tensor():
    """
    Tests the function with tensors of shape [N], which should work.
    Assumes world_size = 2.
    """
    # Assume environment is already set up
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This test is designed to be run with world_size=2"

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Each rank has different data
    if rank == 0:
        # data: [10, 20], indices: [0, 1]
        local_data = torch.tensor([10.0, 20.0], dtype=torch.float32).to(device)
        indices = torch.tensor([0, 1], dtype=torch.long).to(device)
    else:  # rank == 1
        # data: [30, 40], indices: [0, 0]
        local_data = torch.tensor([30.0, 40.0], dtype=torch.float32).to(device)
        indices = torch.tensor([0, 0], dtype=torch.long).to(device)

    # Calculate expected result manually
    # Index 0: 10.0 (r0) + 30.0 (r1) + 40.0 (r1) = 80.0. Count = 3. Mean = 80/3
    # Index 1: 20.0 (r0). Count = 1. Mean = 20/1
    expected_mean = torch.tensor([80.0 / 3.0, 20.0], dtype=torch.float32).to(device)

    # Run the function
    result = distributed_scatter_mean(local_data, indices)

    # Check result
    assert result.shape == expected_mean.shape
    assert torch.allclose(result, expected_mean), (
        f"Expected {expected_mean}, but got {result}"
    )
