from collections.abc import Callable
from typing import override

import torch


class PassLeftCallRight[T_pass, **U_in, U_out]:
    """
    (x, ...z) -> (x, fn(...z))
    """

    def __init__(self, call: Callable[U_in, U_out]) -> None:
        self.call: Callable[U_in, U_out] = call

    def __call__(
        self, left: T_pass, *args: U_in.args, **kwargs: U_in.kwargs
    ) -> tuple[T_pass, U_out]:
        return (left, self.call(*args, **kwargs))


class PassLeftModelRight[T_pass, **U_in, U_out](torch.nn.Module):
    """
    (x, ...z) -> (x, model(...z))
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: torch.nn.Module = model

    @override
    def forward(
        self, left: T_pass, *args: U_in.args, **kwargs: U_in.kwargs
    ) -> tuple[T_pass, U_out]:
        """
        Passes `left` through and applies `self.model` to `*args` and `**kwargs`.
        """
        return (left, self.model(*args, **kwargs))
