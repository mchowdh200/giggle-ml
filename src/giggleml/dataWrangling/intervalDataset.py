import gzip
import random
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import Callable, Protocol, final, override

from ..utils.types import GenomicInterval, lazy
from .kindDataset import KindDataset


class IntervalDataset(Protocol):
    @property
    def associatedFastaPath(self) -> str | None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> GenomicInterval: ...

    def __iter__(self) -> Iterator[GenomicInterval]: ...


@lazy
class LateIntervalDataset(KindDataset[GenomicInterval]):
    """
    An IntervalDataset that treats the backing intervals with a lazy getter
    function. This enables it to be passed to other processes without incurring
    a serialization tax.

    @param lazyLength: None implies inferred from size of lazyGetter()
    """

    def __init__(
        self,
        lazyGetter: Callable[[], list[GenomicInterval]],
        lazyLength: Callable[[], int] | int | None,
        associatedFastaPath: str | None,
    ):
        self.lazyGetter: Callable[[], list[GenomicInterval]] = lazyGetter
        self.lazyLength: Callable[[], int] | int | None = lazyLength
        self.associatedFastaPath: str | None = associatedFastaPath

    @cached_property
    def _content(self):
        return self.lazyGetter()

    @cached_property
    def _length(self):
        if self.lazyLength is None:
            return len(self._content)
        elif isinstance(self.lazyLength, int):
            return self.lazyLength
        else:
            return self.lazyLength()

    @override
    def __len__(self):
        return self._length

    @override
    def __getitem__(self, idx: int):
        return self._content[idx]


@final
class MemoryIntervalDataset(KindDataset[GenomicInterval]):
    def __init__(
        self,
        intervals: Sequence[GenomicInterval],
        associatedFastaPath: str | None = None,
    ):
        super().__init__()
        self.intervals = intervals
        self.associatedFastaPath = associatedFastaPath

    @override
    def __len__(self):
        return len(self.intervals)

    @override
    def __getitem__(self, idx: int) -> GenomicInterval:
        return self.intervals[idx]


@lazy
class BedDataset(LateIntervalDataset):
    def __init__(
        self,
        path: str | Path,
        associatedFastaPath: str | None = None,
        limit: int | None = None,
        samplingRate: float = 1,
    ):
        """
        Bed files are parsed lazily (so this passes pickling barrier).
        @param path: can be either a .bed or .bed.gz file.
        @param samplingRate: should be 0-1, 1 indicates guaranteed full reads. Seed is fixed.
        """
        self.path: str = str(path) if isinstance(path, Path) else path
        assert samplingRate >= 0 and samplingRate <= 1
        self.samplingRate: float = samplingRate
        super().__init__(
            lazyGetter=self._load,
            lazyLength=limit,
            associatedFastaPath=associatedFastaPath,
        )

    def _load(self):
        random.seed(42)

        def parse(file):
            intervals = list[GenomicInterval]()

            for line in file:
                if line.startswith("#"):
                    continue

                if self.samplingRate != 1:
                    if random.random() > self.samplingRate:
                        continue

                name, start, stop = line.split("\t")[:3]
                intervals.append((name, int(start), int(stop)))
            return intervals

        if self.path.endswith(".bed.gz"):
            with gzip.open(self.path, "rt") as file:  # 'rt' mode for text reading
                return parse(file)
        elif self.path.endswith(".bed"):
            with open(self.path, "r") as file:
                return parse(file)
        else:
            raise ValueError("BedDataset expects inputs of either .bed.gz or .bed")
