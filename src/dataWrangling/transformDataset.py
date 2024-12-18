import torch


class Transform:
    def __init__(self, scale=1):
        """
        scale is used to compute the new length (old lengt * scale) and
        to provide capability for the transform to map backing elemnts to multiple new elements.
        It is necesssary to strictly define a single scale because the transformed dataset
        is generated incrementally and the length must be known at init time.
        """
        self.scale = scale

    def __call__(self, item, idx):
        raise NotImplementedError


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, backingDataset, transforms: list[Transform]):
        self.backingDataset = backingDataset
        self.transforms = transforms

        self.scale = 1
        for t in self.transforms:
            self.scale *= t.scale

    def transform(self, backingItem, idx):
        backingIdx = idx / self.scale  # Yes, a float

        for f in self.transforms:
            backingIdx = backingIdx * f.scale
            backingItem = f(backingItem, int(backingIdx))

        return backingItem

    def __len__(self):
        return len(self.backingDataset) * self.scale

    def baseItem(self, idx):
        return self.backingDataset[idx // self.scale]

    def __getitem__(self, idx):
        backingItem = self.baseItem(idx)
        return self.transform(backingItem, idx)
