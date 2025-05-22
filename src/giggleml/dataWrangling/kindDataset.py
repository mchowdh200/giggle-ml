from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override

from torch.utils.data import Dataset


class KindDataset[T](Dataset[T], ABC):
    """
    Provides basic utilities on top of the default pytorch dataset
    """

    # bizzarely, __len__ is actually not defined in Dataset
    @abstractmethod
    def __len__(self) -> int: ...

    @override
    @abstractmethod
    def __getitem__(self, idx: int) -> T: ...

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]
