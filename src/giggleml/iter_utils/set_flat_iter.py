import itertools
from collections.abc import Iterator, Sequence

from giggleml.utils.types import SizedIterable


class SetFlatIter[T]:
    """
    A utility to flatten a nested iterable and provide batched streams
    of the data and corresponding indices.

    This implementation performs a single pass on initialization to learn the
    full structure, caching the data to ensure correctness and consistency
    between its methods.

    All iterator methods (e.g., __iter__, indices) yield batches. Crucially,
    batches *do not* cross the boundaries of the inner sets. If a set ends,
    the current batch also ends, even if it is smaller than batch_size.
    """

    def __init__(
        self,
        data: Sequence[SizedIterable[T]],
        batch_size: int = 1,
    ):
        """
        does not consume the input to learn the structure and cache data.

        Args:
            data: A sequence of sized iterables (e.g., a list of lists).
            batch_size: The desired size for batches. Batches will not
                cross the boundaries of the inner iterables in `data`.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._data: Sequence[SizedIterable[T]] = data
        self._group_lengths: list[int] = [len(group) for group in data]
        self._batch_size: int = batch_size

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        """
        Provides the batched, flattened data stream.
        Iterable[Iterable[T]] -> Iterable[Batch[T]].

        Yields:
            Batches of items as tuples.
        """
        for group in self._data:
            yield from itertools.batched(group, self._batch_size)

    def indices(self) -> Iterator[tuple[tuple[int, int], ...]]:
        """
        Provides batches of the (set index, index within set) mapping.
        These batches correspond directly to the batches yielded by __iter__.

        Yields:
            Batches of (set_index, inner_index) tuples.
        """
        for i, length in enumerate(self._group_lengths):
            # Create a generator for just this group's indices
            # e.g., ((0, 0), (0, 1), (0, 2))
            group_indices = ((i, j) for j in range(length))

            # Batch the indices generator for this group
            yield from itertools.batched(group_indices, self._batch_size)

    def set_indices(self) -> Iterator[tuple[int, ...]]:
        """
        Provides batches of the set indices.
        These batches correspond directly to the batches yielded by __iter__.

        Yields:
            Batches of set_index integers.
        """
        for i, length in enumerate(self._group_lengths):
            # Create a generator for just this group's set indices
            # e.g., (0, 0, 0)
            group_set_indices = (i for _ in range(length))

            # Batch the set indices generator for this group
            yield from itertools.batched(group_set_indices, self._batch_size)
