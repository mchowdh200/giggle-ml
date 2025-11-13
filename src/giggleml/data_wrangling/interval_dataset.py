import gzip
import random
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import Callable, Protocol, final, override

from giggleml.utils.path_utils import as_path

from ..utils.types import GenomicInterval, PathLike, lazy
from .kind_dataset import KindDataset


class IntervalDataset(Protocol):
    @property
    def associated_fasta_path(self) -> Path | None: ...

    def __len__(self) -> int: ...

    # def __getitem__(self, idx: int) -> GenomicInterval: ...

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
        lazy_getter: Callable[[], list[GenomicInterval]],
        lazy_length: Callable[[], int] | int | None,
        associated_fasta_path: PathLike | None,
    ):
        self.lazy_getter: Callable[[], list[GenomicInterval]] = lazy_getter
        self.lazy_length: Callable[[], int] | int | None = lazy_length
        self.associated_fasta_path: Path | None = as_path(associated_fasta_path)

    @cached_property
    def _content(self):
        return self.lazy_getter()

    @cached_property
    def _length(self):
        if self.lazy_length is None:
            return len(self._content)
        elif isinstance(self.lazy_length, int):
            return self.lazy_length
        else:
            return self.lazy_length()

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
        associated_fasta_path: PathLike | None = None,
    ):
        super().__init__()
        self.intervals = intervals
        self.associated_fasta_path = as_path(associated_fasta_path)

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
        path: PathLike,
        associated_fasta_path: PathLike | None = None,
        limit: int | None = None,
        sampling_rate: float = 1,
    ):
        """
        Bed files are parsed lazily (so this passes pickling barrier).
        @param path: can be either a .bed or .bed.gz file.
        @param samplingRate: should be 0-1, 1 indicates guaranteed full reads. Seed is fixed.
        """
        self.path: Path = as_path(path)
        assert sampling_rate >= 0 and sampling_rate <= 1
        self.sampling_rate: float = sampling_rate
        self.limit: float = limit or float("inf")

        super().__init__(
            lazy_getter=self._load,
            lazy_length=None,
            associated_fasta_path=associated_fasta_path,
        )

    def _load(self):
        random.seed(42)

        def parse(file):
            intervals = list[GenomicInterval]()

            for line in file:
                if len(intervals) >= self.limit:
                    break

                if line.startswith("#"):
                    continue

                if self.sampling_rate != 1:
                    if random.random() > self.sampling_rate:
                        continue

                name, start, stop = line.split("\t")[:3]
                intervals.append((name, int(start), int(stop)))
            return intervals

        if self.path.suffixes[-2:] == [".bed", ".gz"]:
            with gzip.open(self.path, "rt") as file:  # 'rt' mode for text reading
                return parse(file)
        elif self.path.suffix == ".bed":
            with open(self.path, "r") as file:
                return parse(file)
        else:
            raise ValueError("BedDataset expects inputs of either .bed.gz or .bed")
