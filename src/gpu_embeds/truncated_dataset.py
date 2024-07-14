from torch.utils.data import Dataset


class TruncatedDataset(Dataset):
    def __init__(self, contents, length):
        self.contents = contents
        self.length = min(len(contents), length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.contents[idx]
