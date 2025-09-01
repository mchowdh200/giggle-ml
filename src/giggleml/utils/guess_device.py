import torch
from torch.types import Device


def guess_device(rank: int = 0) -> Device:
    if torch.accelerator.is_available():
        if torch.backends.mps.is_built():
            return torch.device(f"mps:{rank}")  # for mac use
        if torch.cuda.is_available():
            return torch.device(f"cuda:{rank}")
    return torch.device("cpu")
