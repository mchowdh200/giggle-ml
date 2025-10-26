from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def guess_device(rank: int = 0) -> torch.device:
    if torch.accelerator.is_available():
        if torch.backends.mps.is_built():
            return torch.device(f"mps:{rank}")  # for mac use
        if torch.cuda.is_available():
            return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def freeze_model[T: torch.nn.Module](model: T, unfreeze: bool = False) -> T:
    """modifies the model in place"""

    for param in model.parameters():
        param.requires_grad = unfreeze

    return model


def all_gather_concat(
    tensor: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """
    Gathers tensors of different sizes from all processes and concatenates them.

    In a distributed setting, `torch.distributed.all_gather` requires that the
    tensors on each process have the same shape. This function handles the case
    where the tensors have different sizes along the first dimension.

    The process is as follows:
    1.  Each rank communicates the size of its tensor's first dimension.
    2.  The maximum size among all ranks is determined.
    3.  Each rank pads its tensor to this maximum size.
    4.  A standard `all_gather` operation is performed on the padded tensors.
    5.  The padding is removed from the gathered tensors based on their original sizes.
    6.  The trimmed tensors are concatenated and returned.

    Args:
        tensor (torch.Tensor): The local tensor to be gathered. The shapes must be
                               identical across all ranks except for the first dimension.

    Returns:
        torch.Tensor: A single tensor that is the concatenation of the input
                      tensors from all ranks. This tensor resides on each rank.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    world_size = dist.get_world_size()

    # 1. Gather the size of the first dimension from each rank.
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size, group)

    # 2. Determine the maximum size.
    max_size = int(max(size.item() for size in sizes))

    # 3. Pad the local tensor if its size is less than the max size.
    if local_size.item() < max_size:
        pad_size = int(max_size - local_size.item())
        # Create a padding tensor with the required dimensions.
        # The padding is along the first dimension, other dimensions match the original tensor.
        padding = torch.zeros(
            pad_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
        )
        padded_tensor = torch.cat((tensor, padding), dim=0)
    else:
        padded_tensor = tensor

    # 4. Perform the all_gather on the padded tensors.
    # The output tensor for all_gather needs to be pre-allocated.
    # It will have a shape of (world_size * max_size, ...).
    output_shape = (world_size * max_size, *tensor.shape[1:])
    gathered_padded_tensors = torch.empty(
        output_shape, device=tensor.device, dtype=tensor.dtype
    )

    # PyTorch's all_gather expects a list of tensors for the output when used this way.
    # A more efficient way is to use a single large tensor.
    # Let's reshape for the gather operation.
    # The list will contain tensors that are views into the larger tensor.
    tensor_list = list(gathered_padded_tensors.chunk(world_size, dim=0))
    dist.all_gather(tensor_list, padded_tensor, group)

    # 5. Trim the padding from each tensor in the gathered list.
    result_tensors = []
    for i in range(world_size):
        original_size = sizes[i].item()
        result_tensors.append(tensor_list[i][:original_size])

    # 6. Concatenate the trimmed tensors.
    return torch.cat(result_tensors, dim=0)


def rprint(*args: Any, **kwargs: Any):
    """rank print"""
    rank = get_rank()
    print(f"[rank {rank}]", *args, **kwargs)
