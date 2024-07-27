from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """  Used for the Sliding Window Test.
    Takes sequences from another dataset and returns
    a series of sequences of half-size that overlap
    to various degrees.

    Example:

        (from the backing dataset)
            "The quick brown fox jumps over the lazy dog"

        (this creates...)
            "The quick brown fox j"
             "he quick brown fox ju"
              "e quick brown fox jum"
               " quick brown fox jump"
                            ...
                                 "umps over the lazy dog"

    The overlap between each item and the original item diminishes.
    [----][----]
      ^ original item
            ^ final item:     no overlap

    Gap factor can be used to control the degree of overlap.
    Consider gap factor 10% (.1):
        [----]             <-|
           [----]            |  11 of these (1/gapFactor + 1)
        ^     [----]         |
        |--^     [----]    <-|
        |  |
        10% difference, 90% overlap
    """

    def __init__(self, backingDataset, gapFactor=.1):
        self.backingDataset = backingDataset
        self.cache = (None, None)
        self.spreeCount = int(1 / gapFactor) + 1
        self.gapFactor = gapFactor

    def __len__(self):
        return self.spreeCount * len(self.backingDataset)

    def slide(self, backingIdx: int):
        backingData = self.backingDataset[backingIdx]
        windowSize = len(backingData) // 2

        gapSizeFloat = windowSize * self.gapFactor
        gapSize = int(gapSizeFloat)
        error = 1 - gapSize / gapSizeFloat

        errThreshold = .05
        if error > errThreshold:
            raise ValueError(
                f"Gap size error above {errThreshold}. Backing sequence is too short.", error)

        for i in range(0, len(backingData) - windowSize, gapSize):
            left = i
            right = i + windowSize
            yield backingData[left:right]

    def __getitem__(self, idx):
        backingIdx = idx // self.spreeCount

        if self.cache[0] != backingIdx:
            nextSet = list(self.slide(backingIdx))
            self.cache = (backingIdx, nextSet)

        backingData = self.cache[1]
        offset = idx % self.spreeCount
        return backingData[offset]


# TODO: easy unit test
#
# if __name__ == "__main__":
#     content = [
#         "When is the next train to London?",
#         "The quick brown fox jumps over the lazy dog",
#         "All work and no play makes Jack a dull boy",
#         "123456789"
#     ]
#
#     dataset = SlidingWindowDataset(content, .75)
#
#     for i in range(len(dataset)):
#         x = dataset[i]
#         print(x)
#
#     # When is the next
#     # next train to Lo
#     # The quick brown fox j
#     # fox jumps over the la
#     # All work and no play
#     # play makes Jack a dul
#     # 1234
#     # 4567
