from abc import ABC
from collections.abc import Iterable
from typing import Any, final, override

import numpy as np

from giggleml.utils.intervalArithmetic import intersect

from .utils.types import GenomicInterval


class IntervalTransform(ABC):
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]: ...


@final
class ChunkMax(IntervalTransform):
    def __init__(self, maxLen: int):
        self.maxLen: int = maxLen

    @override
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        intLen = interval[2] - interval[1]

        for start in range(0, intLen, self.maxLen):
            end = min(start + self.maxLen, intLen)
            yield ((interval[0], interval[1] + start, interval[1] + end))


@final
class Translate(IntervalTransform):
    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start, end = item
        yield (chrm, start + self.offset, end + self.offset)


@final
class Swell(IntervalTransform):
    def __init__(self, swellFactor: float = 0.5):
        super().__init__()
        self.swellFactor = swellFactor

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start0, end0 = item
        size0 = end0 - start0
        size = round(self.swellFactor * size0)
        start = round((start0 + end0) / 2 - size / 2)
        yield chrm, start, start + size


@final
class Split(IntervalTransform):
    def __init__(self, factor: int = 3):
        super().__init__()
        if factor < 1:
            raise ValueError("Split factor must be at least 1")
        self.factor = factor

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start, end = item
        size = end - start

        if size < self.factor:
            yield item
            return

        base, remainder = divmod(size, self.factor)
        start2 = start

        for i in range(self.factor):
            size2 = base + 1 if i < remainder else base
            end2 = start2 + size2
            yield (chrm, start2, end2)
            start2 = end2


@final
class Slide(IntervalTransform):
    """
    Eg,

    steps = 4,
    original interval:  |-------|
    yields:             |-------|
                          |-------|
                             |-------|
                                |-------|
                        33%  ^--^
    33% corresponds to strideNumber=3 as in 1/strideNumber overlap% each step

    where the first interval has an overlap of 1,
    the last interval has an overlap of 0,
    and the spacing between intervals is (original size) / (steps-1)

    The direction to slide: originalInterval.start % 2 == 0 --> right
    """

    def __init__(self, steps: int = 11, strideNumber: int | None = None):
        """
        @param strideNumber: defaults to (steps-1)
        """

        super().__init__()
        self.steps = steps
        self.strideNumber = strideNumber or (steps - 1)

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start, end = item
        size = end - start
        stride = size / self.strideNumber

        if start % 2 == 1:
            stride *= -1

        for i in range(self.steps):
            start2 = round(start + i * stride)
            end2 = round(end + i * stride)
            yield chrm, start2, end2


@final
class Tiling(IntervalTransform):
    """
    Simulates a tiling. Implemented for a new embedding procedure where intervals
    are broken down into representative tiles that would have been pre-embedded.
    """

    def __init__(self, tileSize: int):
        self.tileSize = tileSize

    @override
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        startTile = interval[1] // self.tileSize
        endTile = (interval[2] - 1) // self.tileSize

        for tileIdx in range(startTile, endTile + 1):
            start = tileIdx * self.tileSize
            yield (interval[0], start, start + self.tileSize)

    def weights(
        self, intervals: Iterable[GenomicInterval]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        weights = list()

        for base in intervals:
            terms = self(base)
            chunk = list()
            chunkWeight = 0

            for term in terms:
                overlap = intersect(term, base)
                assert overlap is not None
                amnt = overlap[2] - overlap[1]
                chunkWeight += amnt
                chunk.append(amnt)

            slice = np.array(chunk, np.float32) / chunkWeight
            weights.append(slice)

        return np.array(weights, np.float32).reshape(-1)
