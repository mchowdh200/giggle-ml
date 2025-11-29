import os
from functools import cache
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lightning_fabric.fabric import Fabric
from lightning_fabric.loggers.tensorboard import TensorBoardLogger


def is_distributed() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def guess_device(rank: int | None = None) -> torch.device:
    rank = rank if rank is not None else get_rank()

    if hasattr(torch, "accelerator"):
        if torch.accelerator.is_available():
            if (dev := torch.accelerator.current_accelerator()) is not None:
                return dev
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_built():
        return torch.device("mps")  # for mac use
    else:
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


def rprint0(*args: Any, **kwargs: Any):
    """rank print only if rank zero"""
    if get_rank() == 0:
        rprint(*args, **kwargs)


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        # Get the first parameter from the generator
        return next(module.parameters()).device
    except StopIteration:
        # This handles the case for parameter-less modules
        return torch.device("cpu")


@cache
def launch_fabric():
    logger = TensorBoardLogger(root_dir="logs")
    world_size = int(os.environ.get("WORLD_SIZE") or 1)
    fabric = Fabric(
        accelerator="auto",
        strategy="auto",
        devices=world_size,
        loggers=[logger],
    )
    fabric.launch()
    fabric.seed_everything(42)
    dist.barrier()
    return fabric


def torch2numpy_dtype(dt: torch.dtype) -> np.dtype:
    """Convert PyTorch dtype to NumPy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.dtype("V2"),  # No direct NumPy equivalent
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }

    if dt not in mapping:
        raise ValueError(f"Unsupported PyTorch dtype: {dt}")

    return mapping[dt]  # pyright: ignore[reportReturnType]
