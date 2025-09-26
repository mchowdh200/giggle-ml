import torch
from torch.types import Device


def guess_device(rank: int = 0) -> Device:
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
