import torch


# operates on bed files
class SwellChunkDataset(torch.utils.data.Dataset):
    def __init__(self, bedDs, swellFactor=0.5, chunkAmnt=3):
        self.bedDs = bedDs
        self.chunkAmnt = chunkAmnt
        self.swellFactor = swellFactor

    def __len__(self):
        return len(self.bedDs) * self.chunkAmnt

    def swell(self, interval):
        size = interval[2] - interval[1]
        delta = int(self.swellFactor * size // 2)
        return (interval[0], interval[1] - delta, interval[2] + delta)

    def baseIdx(self, idx):
        return idx // self.chunkAmnt

    def baseItem(self, idx):
        """
        idx is with respect to this dataset; it is not the base index.
        """
        return self.bedDs[self.baseIdx(idx)]

    def __getitem__(self, idx):
        # TODO: operations on the base item can be re-used
        baseItem = self.swell(self.baseItem(idx))
        chunkSize = (baseItem[2] - baseItem[1]) // self.chunkAmnt
        chunkIdx = idx % self.chunkAmnt

        chrm = baseItem[0]
        start = baseItem[1] + chunkIdx * chunkSize
        end = start + chunkSize

        return (chrm, start, end)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
