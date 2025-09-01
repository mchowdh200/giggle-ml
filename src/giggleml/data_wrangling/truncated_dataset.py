from typing import final, override

from torch.utils.data import Dataset

from giggleml.utils.types import ListLike


@final
class TruncatedDataset[T](Dataset):
    def __init__(self, contents: ListLike[T], length: int):
        self.contents = contents
        if length:
            self.length = min(len(contents), length)
        else:
            self.length = len(contents)

    def __len__(self):
        return self.length

    @override
    def __getitem__(self, idx: int) -> T:
        return self.contents[idx]
