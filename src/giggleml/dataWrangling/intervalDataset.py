import gzip
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from typing import Callable, final, override

from torch.utils.data import Dataset
from typing_extensions import override

from ..utils.types import GenomicInterval, lazy


class IntervalDataset(Dataset[GenomicInterval], ABC):
    associatedFastaPath: str | None

    # bizzarely, __len__ is actually not defined in Dataset
    @abstractmethod
    def __len__(self) -> int: ...

    @override
    @abstractmethod
    def __getitem__(self, idx: int) -> GenomicInterval: ...

    def __iter__(self) -> Iterator[GenomicInterval]:
        for i in range(len(self)):
            yield self[i]


@lazy
@final
class LateIntervalDataset(IntervalDataset):
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
        self.lazyGetter = lazyGetter
        self.lazyLength = lazyLength
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
class MemoryIntervalDataset(IntervalDataset):
    def __init__(
        self, intervals: Sequence[GenomicInterval], associatedFastaPath: str | None = None
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
        path: str,
        associatedFastaPath: str | None = None,
    ):
        """
        Bed files are parsed lazily (so this passes pickling barrier).
        @param path: can be either a .bed or .bed.gz file.
        """
        self.path: str = path
        super().__init__(
            lazyGetter=self._load, lazyLength=None, associatedFastaPath=associatedFastaPath
        )

    def _load(self):
        def parse(file):
            intervals = list[GenomicInterval]()

            for line in file:

                if line.startswith("#"):
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
