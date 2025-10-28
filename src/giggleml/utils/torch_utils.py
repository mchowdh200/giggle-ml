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


def guess_device(rank: int | None = None) -> torch.device:
    rank = rank or get_rank()

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


def rprint(*args: Any, **kwargs: Any):
    """rank print"""
    rank = get_rank()
    print(f"[rank {rank}]", *args, **kwargs)


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        # Get the first parameter from the generator
        return next(module.parameters()).device
    except StopIteration:
        # This handles the case for parameter-less modules
        return torch.device("cpu")
