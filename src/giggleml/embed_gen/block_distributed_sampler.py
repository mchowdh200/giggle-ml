from typing import Sized, final

from torch.utils.data import Sampler
from typing_extensions import override


@final
class BlockDistributedSampler(Sampler):
    """
    Deterministically splits a dataset into contiguous blocks for each replica.
    Does not require communication between replicas.
    """

    def __init__(self, dataset: Sized, num_replicas=None, rank=None):
        if num_replicas is None or rank is None:
            raise ValueError("num_replicas and rank must be provided")
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("num_replicas should be a positive integer")
        if not isinstance(rank, int) or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the range" f" [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)

        if self.total_size == 0:
            self.lower = 0
            self.upper = 0
        elif self.total_size < self.num_replicas:
            # Assign first totalSize ranks one sample each, others get zero
            self.lower = self.rank if self.rank < self.total_size else self.total_size
            self.upper = self.rank + 1 if self.rank < self.total_size else self.total_size
        else:
            # Standard case
            small = self.total_size // self.num_replicas
            self.lower = small * self.rank
            # The last rank takes the remainder
            self.upper = (
                self.total_size if self.rank == self.num_replicas - 1 else small * (self.rank + 1)
            )

    @override
    def __iter__(self):
        # Use pre-calculated bounds
        return iter(range(self.lower, self.upper))

    def __len__(self):
        # Return pre-calculated number of samples for this rank
        return self.upper - self.lower


# Testig purposes
# class TrivialDataset(Dataset):
#     def __init__(self, size=9):
#         self.size = size
#         self.data = torch.arange(size)
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#
# # Create an instance of the dataset
# ds = TrivialDataset()
#
# for x in ds:
#     print(x)
# print()
# print()
# print()
#
# sa1 = BlockDistributedSampler(ds, num_replicas=2, rank=0)
# sa2 = BlockDistributedSampler(ds, num_replicas=2, rank=1)
#
# for x in sa1:
#     print(x)
# print()
# for x in sa2:
#     print(x)
# print()
# print()
# print()
#
# dl = DataLoader(ds, batch_size=2, sampler=sa2, shuffle=False)
#
# for x in dl:
#     print(x)
# print()
