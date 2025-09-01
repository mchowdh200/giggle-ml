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
        running_total = 0

        for item in lists:
            self.sums.append(running_total)
            size = len(item)
            running_total += size

        self.sums.append(running_total)
        self.len = self.sums[-1]
        self.previous_special_idx: int = 0

    def __len__(self):
        return self.len

    def list_idx_of(self, idx: int):
        if idx >= len(self):
            raise ValueError("idx exceeds length")

        def binary_search():
            l, r = 0, len(self.sizes)

            while l + 1 < r:
                center_idx = (r + l) // 2
                center = self.sums[center_idx]

                if center <= idx:
                    l = center_idx
                elif center > idx:
                    r = center_idx
            return l

        prev = self.previous_special_idx

        # optimization for case of repeat calls with increasing idx
        if self.sums[prev] <= idx:
            offset = idx - self.sums[prev]
            prev_len = self.sizes[prev]

            if offset < prev_len:
                l = prev
            elif offset == prev_len:
                l = prev + 1
            else:
                l = binary_search()
        else:
            l = binary_search()

        # the list that contains idx
        self.previous_special_idx = l
        return l

    @override
    def __getitem__(self, idx: int) -> T:
        i = self.list_idx_of(idx)
        offset = idx - self.sums[i]
        special_list = self.lists[i]
        return special_list[offset]
