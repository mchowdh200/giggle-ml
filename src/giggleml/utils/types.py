from collections.abc import Iterable, Sized
from typing import Any, Protocol, TypeVar

import numpy as np

GenomicInterval = tuple[str, int, int]
MmapF32 = np.memmap[Any, np.dtype[np.float32]]


def lazy[T](cls: T) -> T:
    """
    This decorator is only an annotation -- it provides no functionality.

    Marks an object as "lazy" indicating that any members within it, and the
    computation associated with them, will not occur until accessed. This allows
    it to be passed through pickling barriers without incurring the tax.
    """
    return cls


TCo = TypeVar("T_co", covariant=True)


class ListLike[TCo](Protocol):
    def __getitem__(self, idx: int) -> TCo: ...
    def __len__(self) -> int: ...


class SizedIterable[T](Sized, Iterable[T], Protocol):
    pass
