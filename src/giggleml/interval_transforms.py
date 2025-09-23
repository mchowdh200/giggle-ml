from abc import ABC
from collections.abc import Iterable
from math import floor, log2
from typing import Any, final, override

import numpy as np

from giggleml.utils.interval_arithmetic import overlap_degree

from .utils.types import GenomicInterval


class IntervalTransform(ABC):
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]: ...


@final
class ChunkMax(IntervalTransform):
    def __init__(self, max_len: int):
        self.max_len: int = max_len

    @override
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        int_len = interval[2] - interval[1]

        for start in range(0, int_len, self.max_len):
            end = min(start + self.max_len, int_len)
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
    def __init__(self, swell_factor: float = 0.5):
        super().__init__()
        self.swell_factor = swell_factor

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start0, end0 = item
        size0 = end0 - start0
        size = round(self.swell_factor * size0)
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

    def __init__(self, steps: int = 11, stride_number: int | None = None):
        """
        @param strideNumber: defaults to (steps-1)
        """

        super().__init__()
        self.steps = steps
        self.stride_number = stride_number or (steps - 1)

    @override
    def __call__(self, item: GenomicInterval) -> Iterable[GenomicInterval]:
        chrm, start, end = item
        size = end - start
        stride = size / self.stride_number

        if start % 2 == 1:
            stride *= -1

        for i in range(self.steps):
            start2 = round(start + i * stride)
            end2 = round(end + i * stride)
            yield chrm, start2, end2


@final
class Tiling(IntervalTransform):
    def __init__(
        self,
        tile_size: int,
        octaves: int = 1,
        offsets: int = 1,
    ):
        if octaves < 1:
            raise ValueError(f"Expecting octaves ({octaves}) >= 1")

        self.octave_amnt = octaves
        self._base_size = tile_size

        if offsets < 1:
            raise ValueError(f"Expecting at least 1 offset, got {offsets}")
        self.offsets = offsets

    def _tile(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        """
        Recursively tiles an interval using the Greedy Midpoint Snapping algorithm.
        """
        chrm, start, end = interval
        length = end - start

        # --- STEP 1: SELECT TILE SIZE ---
        if length < self._base_size:
            octave_size = self._base_size
        else:
            octave_idx = floor(log2(length / self._base_size))
            octave_idx = min(octave_idx, self.octave_amnt - 1)
            octave_size = self._base_size * (2**octave_idx)

            if octave_idx < 0:
                return

        # --- STEP 2: FIND BEST POSITION (THE "SNAP") ---
        interval_mid = start + length / 2
        midpoint_grid_spacing = octave_size / self.offsets
        snapped_midpoint = (
            round(interval_mid / midpoint_grid_spacing) * midpoint_grid_spacing
        )

        # --- STEP 3: PLACE TILE ---
        tile_start = round(snapped_midpoint - octave_size / 2)
        tile_end = tile_start + octave_size
        tile = (chrm, tile_start, tile_end)

        # --- Don't yield small tiles with <50% overlap ---
        overlap_length = overlap_degree(interval, tile)

        if overlap_length < (tile_end - tile_start) / 2:
            return

        # Yield the greedily chosen best-fit tile
        yield (chrm, tile_start, tile_end)

        # --- STEP 4: RECURSE on the tips ---
        # Left tip
        left_tip_interval = (chrm, start, tile_start)
        yield from self._tile(left_tip_interval)

        # Right tip
        right_tip_interval = (chrm, tile_end, end)
        yield from self._tile(right_tip_interval)

    @override
    def __call__(self, interval: GenomicInterval) -> Iterable[GenomicInterval]:
        """
        Generates the set of tiles covering the interval using the greedy algorithm.
        """
        yield from self._tile(interval)

    def weights(
        self, intervals: Iterable[GenomicInterval]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        weights = list()

        for base in intervals:
            terms = self(base)
            chunk = list()
            chunk_weight = 0

            for term in terms:
                amnt = overlap_degree(term, base)
                chunk_weight += amnt
                chunk.append(amnt)

            slice = np.array(chunk, np.float32) / chunk_weight
            weights.append(slice)

        return np.array(weights, np.float32).reshape(-1)
