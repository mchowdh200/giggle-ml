import gzip
from abc import ABC
from functools import cached_property
from typing import Callable

from torch.utils.data import Dataset
from typing_extensions import override

from ..utils.types import GenomicInterval, lazy


class IntervalDataset(Dataset[GenomicInterval], ABC):
    associatedFastaPath: str | None

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> GenomicInterval: ...


@lazy
class LateIntervalDataset(IntervalDataset):
    """
    An IntervalDataset that treats the backing intervals with a lazy getter
    function. This enables it to be passed to other processes without incurring
    a serialization tax.
    """

    def __init__(
        self,
        lazyGetter: Callable[[], list[GenomicInterval]],
        associatedFastaPath: str | None,
    ):
        self.lazyGetter: Callable[[], list[GenomicInterval]] = lazyGetter
        self.associatedFastaPath: str | None = associatedFastaPath

    @cached_property
    def _content(self):
        return self.lazyGetter()

    @override
    def __len__(self):
        # bizzarely, __len__ is actually not defined in Dataset
        return len(self._content)

    @override
    def __getitem__(self, idx: int):
        return self._content[idx]


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
        super().__init__(lazyGetter=self._load, associatedFastaPath=associatedFastaPath)

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
