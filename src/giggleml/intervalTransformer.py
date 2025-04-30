import os
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import Protocol, final

import numpy as np

from .dataWrangling.intervalDataset import IntervalDataset, LateIntervalDataset
from .utils.types import GenomicInterval, lazy

# =================================
#          Transforms
# =================================


class IntervalTransform(Protocol):
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]: ...


class ChunkMax:
    def __init__(self, maxLen: int):
        self.maxLen: int = maxLen

    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        intLen = interval[2] - interval[1]

        for start in range(0, intLen, self.maxLen):
            end = min(start + self.maxLen, intLen)
            yield ((interval[0], interval[1] + start, interval[1] + end))


# =================================
#     Interval Transformer
# =================================


@final
@lazy
class IntervalTransformer:
    """
    Maintains a mapping between an old IntervalDataset, based on a series
    of transformations used to generate new intervals, and a new
    IntervalDataset. The Transform(s) from an old interval can create new
    intervals or delete intervals, but not merge intervals. Thus each element in
    the new dataset maps to exactly one base element.
    """

    def __init__(
        self,
        oldDataset: IntervalDataset,
        transforms: Sequence[IntervalTransform] | IntervalTransform,
    ):
        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms = transforms
        self.oldDataset = oldDataset

        self._built: bool = False
        self._newIntervals: list[GenomicInterval]
        self._toIdx: list[list[int]]
        self._fromIdx: list[int]

    def transform(self, interval: GenomicInterval, i: int = 0) -> Iterable[GenomicInterval]:
        if i >= len(self.transforms):
            yield interval
        else:
            newIntervals = self.transforms[i](interval)
            for interval in newIntervals:
                for newInterval in self.transform(interval, i + 1):
                    yield newInterval

    def build(self):
        if self._built:
            return

        results = list()
        toMap: list[list[int]] = [list()] * len(self.oldDataset)
        fromMap: list[int] = list()

        for i in range(len(self.oldDataset)):
            oldInterval = self.oldDataset[i]
            newIntervals = list(self.transform(oldInterval))
            j = len(results)
            toMap[i] = list(range(j, j + len(newIntervals)))
            fromMap.extend([i] * len(newIntervals))
            results.extend(newIntervals)

        self._newIntervals = results
        self._toIdx = toMap
        self._fromIdx = fromMap
        self._built = True

    def getNewIntervals(self):
        """
        This method only exists as an accessible method for the
        LateIntervalDataset to call when preparing it's internal data. It could
        not be a cached_property because the dataset requires a function
        pointer.
        """
        self.build()
        return self._newIntervals

    @cached_property
    def newDataset(self) -> LateIntervalDataset:
        return LateIntervalDataset(self.getNewIntervals, self.oldDataset.associatedFastaPath)

    @cached_property
    def toIdx(self):
        self.build()
        return self._toIdx

    @cached_property
    def fromIdx(self):
        self.build()
        return self._fromIdx

    def backwardTransform(
        self, memmap: np.memmap, aggregator: Callable[[np.ndarray], np.ndarray]
    ) -> np.memmap:
        """
        Maps items from the memmap located at outPath back to
        a memmap, the same length as oldDataset, using the aggregator
        function to combine elements. This function is generic to the
        memmap datatype. It will rebuild the file associated with the memmap
        passed in and return a new memmap pointing to the file after completion.

        This is NOT a machine learning architecture.

        @param aggregator: Operates on slices from the mmap[T]. If memmap is NxD
        then for an arbitrary slice size K, the aggregator would be used in the
        form [KxD] -> [1xD]
        """

        frontPath = memmap.filename

        if type(frontPath) is not str:
            raise ValueError("Unable to determine memmap filename.")

        if len(memmap) != len(self.newDataset):
            raise ValueError("Memmap must be equal in length to intervalTransformer.newDataset")

        backPath = frontPath + ".temp"
        backShape = memmap.shape
        backShape = (len(self.oldDataset), backShape[1])
        dtype = memmap.dtype
        backMem = np.memmap(backPath, dtype=dtype, mode="w+", shape=backShape)

        j = 0
        for i in range(len(self.oldDataset)):
            frontCount = len(self.toIdx[i])
            slice = memmap[j : j + frontCount]
            aggregate = aggregator(slice)
            j += frontCount
            backMem[i] = aggregate

        backMem.flush()
        del backMem
        memmap.flush()
        del memmap

        os.remove(frontPath)
        os.rename(backPath, frontPath)

        return np.memmap(frontPath, dtype=dtype, mode="r", shape=backShape)
