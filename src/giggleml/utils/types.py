from typing import Any, Protocol, SupportsIndex, TypeVar

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


T_co = TypeVar("T_co", covariant=True)


class ListLike[T_co](Protocol):
    def __getitem__(self, idx: int) -> T_co: ...
    def __len__(self) -> int: ...
