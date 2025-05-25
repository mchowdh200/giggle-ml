from collections.abc import Sequence
from typing import final, override

from torch.utils.data import Dataset

from giggleml.utils.types import ListLike


@final
class UnifiedDataset[T](Dataset):
    """
    Allows a series of lists to be used as if combined in a single list.
    """

    def __init__(self, lists: Sequence[ListLike[T]]):
        self.lists = lists
        self.sizes = [len(item) for item in lists]

        self.sums = list[int]()
        runningTotal = 0

        for item in lists:
            self.sums.append(runningTotal)
            size = len(item)
            runningTotal += size

        self.sums.append(runningTotal)
        self.len = self.sums[-1]
        self.previousSpecialIdx: int = 0

    def __len__(self):
        return self.len

    def listIdxOf(self, idx: int):
        if idx >= len(self):
            raise ValueError("idx exceeds length")

        def binarySearch():
            l, r = 0, len(self.sizes)

            while l + 1 < r:
                centerIdx = (r + l) // 2
                center = self.sums[centerIdx]

                if center <= idx:
                    l = centerIdx
                elif center > idx:
                    r = centerIdx
            return l

        prev = self.previousSpecialIdx

        # optimization for case of repeat calls with increasing idx
        if self.sums[prev] <= idx:
            offset = idx - self.sums[prev]
            prevLen = self.sizes[prev]

            if offset < prevLen:
                l = prev
            elif offset == prevLen:
                l = prev + 1
            else:
                l = binarySearch()
        else:
            l = binarySearch()

        # the list that contains idx
        self.previousSpecialIdx = l
        return l

    @override
    def __getitem__(self, idx: int) -> T:
        i = self.listIdxOf(idx)
        offset = idx - self.sums[i]
        specialList = self.lists[i]
        return specialList[offset]
