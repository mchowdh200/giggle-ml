from collections.abc import Sequence
from typing import final, override

from torch.utils.data import Dataset


@final
class ListDataset[T](Dataset[T]):
    def __init__(self, data: Sequence[T]):
        self.data = data

    def __len__(self):
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> T:
        return self.data[idx]
