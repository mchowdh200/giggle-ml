import math
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

    def __init__(self, tileSize: int, octaves: int = 1):
        """
        @param octaves: the second octave is twice the size as the fundamental
        """

        if octaves < 1:
            raise ValueError(f"Expecting octaves ({octaves}) >= 1")

        self.octaveAmnt = octaves
        self.tileSizes = [tileSize * int(2**i) for i in range(octaves)]

    def tile(self, interval: GenomicInterval, level: int | None = None) -> Iterable[list[int]]:
        chrm, start, end = interval

        level = self.octaveAmnt - 1 if level is None else level
        assert level < self.octaveAmnt

        if end - start == 0:
            yield from [list() for _ in range(level + 1)]
            return

        if level > 0:
            # --- case 1)  greedily take biggest tiles -within- the target
            octaveSize = self.tileSizes[level]
            # round start/end to fundamental to give larger octaves more opportunity,
            #  but don't round start/end in general so that the final layer can
            #  interpolate properly
            roundStart = round(start / self.tileSizes[0]) * self.tileSizes[0]
            roundEnd = round(end / self.tileSizes[0]) * self.tileSizes[0]
            startIdx = math.ceil(roundStart / octaveSize)
            endIdx = math.floor(roundEnd / octaveSize)

            if startIdx >= endIdx:
                yield from self.tile(interval, level - 1)
                yield list()
                return

            # attempt to tile the tips with smaller tiles
            for list1, list2 in zip(
                self.tile((chrm, start, startIdx * octaveSize), level - 1),
                self.tile((chrm, endIdx * octaveSize, end), level - 1),
            ):
                yield list1 + list2

            yield list(range(startIdx, endIdx))
        else:
            # --- case 2)  force fill the remainder with smallest tiles
            octaveSize = self.tileSizes[level]
            startIdx = start // octaveSize
            endIdx = (end - 1) // octaveSize
            yield list(range(startIdx, endIdx + 1))

    @override
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, _, _ = interval

        for i, chunks in enumerate(self.tile(interval)):
            octaveSize = self.tileSizes[i]

            for tileIdx in chunks:
                tileStart = tileIdx * octaveSize
                yield (chrm, tileStart, tileStart + octaveSize)

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
