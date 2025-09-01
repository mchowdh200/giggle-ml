import os
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import final, overload

import numpy as np

from giggleml.interval_transforms import IntervalTransform

from .data_wrangling.interval_dataset import IntervalDataset, LateIntervalDataset
from .utils.types import GenomicInterval, lazy


@final
@lazy
class IntervalTransformer:
    """
    Maintains a mapping between an old IntervalDataset, based on a series
    of transformations used to generate new intervals, and a new
    IntervalDataset. The Transform(s) from an old interval can create new
    intervals or delete intervals, but not merge intervals. Thus each element in
    the new dataset maps to exactly one base element.

    @param lengthLimit: can be used to instill an arbitrary length limit to the
    oldDataset
    """

    def __init__(
        self,
        old_dataset: IntervalDataset,
        transforms: Sequence[IntervalTransform] | IntervalTransform,
        length_limit: int | None = None,
    ):
        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms = transforms
        self.old_dataset = old_dataset

        self._old_length = (
            min(length_limit, len(old_dataset))
            if length_limit is not None
            else len(old_dataset)
        )

        self._built: bool = False
        self._new_intervals: list[GenomicInterval]
        self._to_idx: list[list[int]]
        self._from_idx: list[int]

    def transform(
        self, interval: GenomicInterval, i: int = 0
    ) -> Iterable[GenomicInterval]:
        if i >= len(self.transforms):
            yield interval
        else:
            new_intervals = self.transforms[i](interval)
            for interval in new_intervals:
                for new_interval in self.transform(interval, i + 1):
                    yield new_interval

    def build(self):
        if self._built:
            return

        results = list()
        to_map: list[list[int]] = [list()] * self._old_length
        from_map: list[int] = list()

        for i in range(self._old_length):
            old_interval = self.old_dataset[i]
            new_intervals = list(self.transform(old_interval))
            j = len(results)
            to_map[i] = list(range(j, j + len(new_intervals)))
            from_map.extend([i] * len(new_intervals))
            results.extend(new_intervals)

        self._new_intervals = results
        self._to_idx = to_map
        self._from_idx = from_map
        self._built = True

    @cached_property
    def _length(self):
        total = 0

        for i in range(self._old_length):
            interval = self.old_dataset[i]
            growth = len(list(self.transform(interval)))
            total += growth

        return total

    def __len__(self):
        """
        The length of the newDataset. Can be used to compute it's length in a
        slightly less expensive way than outright calling build(.) or
        len(.newDataset)
        """
        return self._length

    def get_new_intervals(self):
        """
        This method only exists as an accessible method for the
        LateIntervalDataset to call when preparing it's internal data. It could
        not be a cached_property because the dataset requires a function
        pointer.
        """
        self.build()
        return self._new_intervals

    @cached_property
    def new_dataset(self) -> LateIntervalDataset:
        return LateIntervalDataset(
            self.get_new_intervals, self.__len__, self.old_dataset.associated_fasta_path
        )

    def forward_idx(self, old_idx: int):
        """
        Convert an oldDataset index forward to the newDataset
        """
        self.build()
        return self._to_idx[old_idx]

    def backward_idx(self, new_idx: int):
        """
        Convert a newDataset index backward to the oldDataset
        """
        self.build()
        return self._from_idx[new_idx]

    @overload
    def backward_transform(
        self, data: np.memmap, aggregator: Callable[[np.ndarray], np.ndarray]
    ) -> np.memmap: ...

    @overload
    def backward_transform(
        self, data: np.ndarray, aggregator: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray: ...

    def backward_transform(
        self,
        data: np.memmap | np.ndarray,
        aggregator: Callable[[np.ndarray], np.ndarray],
    ) -> np.memmap | np.ndarray:
        """
        Maps items from the data (memmap or numpy array) back to
        a result of the same length as oldDataset, using the aggregator
        function to combine elements.

        For memmap: rebuilds the file and returns a new memmap.
        For numpy array: operates in-memory and returns a new numpy array.

        This is NOT a machine learning architecture.

        @param data: Either a memmap or numpy array to transform backward
        @param aggregator: Operates on slices from the data[T]. If data is NxD
        then for an arbitrary slice size K, the aggregator would be used in the
        form [KxD] -> [1xD]
        """

        if len(data) != len(self.new_dataset):
            raise ValueError(
                "Data must be equal in length to intervalTransformer.newDataset"
            )

        # we are about to access _toIdx
        self.build()

        if isinstance(data, np.memmap):
            # Original memmap implementation
            front_path = data.filename

            if type(front_path) is not str:
                raise ValueError("Unable to determine memmap filename.")

            back_path = front_path + ".temp"
            back_shape = data.shape
            back_shape = (self._old_length, back_shape[1])
            dtype = data.dtype
            back_mem = np.memmap(back_path, dtype=dtype, mode="w+", shape=back_shape)

            data.flush()
            j = 0

            for i in range(self._old_length):
                front_count = len(self._to_idx[i])
                slice = data[j : j + front_count]
                aggregate = aggregator(slice)
                j += front_count
                back_mem[i] = aggregate

            back_mem.flush()
            del back_mem
            del data

            os.remove(front_path)
            os.rename(back_path, front_path)

            return np.memmap(front_path, dtype=dtype, mode="r", shape=back_shape)

        else:
            # In-memory numpy array implementation
            back_shape = data.shape
            back_shape = (self._old_length, back_shape[1])
            back_array = np.empty(back_shape, dtype=data.dtype)

            j = 0
            for i in range(self._old_length):
                front_count = len(self._to_idx[i])
                slice = data[j : j + front_count]
                aggregate = aggregator(slice)
                j += front_count
                back_array[i] = aggregate

            return back_array
