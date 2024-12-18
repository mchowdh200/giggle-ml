from torch.utils.data import Sampler


class BlockDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None or rank is None:
            raise ValueError("num_replicas and rank must be set")
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank %d, rank should be in the range [0, num_replicas)" % rank)

        self.dataset = dataset
        self.numReplicas = num_replicas
        self.rank = rank

        self.totalSize = len(self.dataset)
        self.rankSize = self.totalSize // self.numReplicas

    def __iter__(self):
        upper = None
        if self.rank == self.numReplicas - 1:
            upper = self.totalSize
        else:
            upper = self.rankSize * (self.rank + 1)
        lower = self.rankSize * self.rank
        return iter(range(lower, upper))

    def __len__(self):
        if self.rank == self.numReplicas - 1:
            return self.totalSize - self.rankSize * self.rank
        return self.rankSize


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
