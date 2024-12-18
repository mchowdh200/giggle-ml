from dataWrangling.transformDataset import Transform


class Translate(Transform):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def __call__(self, item, idx):
        chrm, start, end = item
        return (chrm, start + self.offset, end + self.offset)


class Swell(Transform):
    def __init__(self, swellFactor=0.5):
        super().__init__()
        self.swellFactor = swellFactor

    def __call__(self, item, idx):
        chrm, start, end = item
        size = end - start
        delta = int((self.swellFactor - 1) * size // 2)
        return (chrm, start - delta, end + delta)


class Chunk(Transform):
    def __init__(self, chunkAmnt=3):
        super().__init__(scale=chunkAmnt)

    def __call__(self, item, idx):
        chrm, start, end = item
        chunkSize = (end - start) // self.scale

        if chunkSize < 1:
            raise ValueError(
                "Chunk length {} is too small to be divided.".format(end-start))

        chunkIdx = idx % self.scale
        start = start + chunkIdx * chunkSize

        # last chunk gets the remainder
        if chunkIdx != self.scale - 1:
            end = start + chunkSize

        return (chrm, start, end)


class VaryLength(Transform):
    def __init__(self, generator):
        # generator: (idx) -> float
        super().__init__()
        self.generator = generator

    def __call__(self, item, idx):
        chrm, start, end = item
        size = end - start
        delta = int(self.generator(idx) * size)
        return (chrm, start, end + delta)
