from collections.abc import Iterator, Sequence
from math import ceil

from giggleml.utils.types import SizedIterable


class SetFlatIter[T]:
    """
    A utility to flatten a nested iterable and later regroup a
    corresponding flat iterable back into the original structure.

    This implementation performs a single pass on initialization to learn the
    full structure, caching the data to ensure correctness and consistency
    between its methods.
    """

    def __init__(self, data: Sequence[SizedIterable[T]], round_to_multiple: int = 1):
        """
        Initializes and consumes the input to learn the structure and cache data.
        @arg round_to_multiple: rounds each set length to this nearest multiple, filling Nones as necessary
        """
        self._data: Sequence[SizedIterable[T]] = data
        self._group_lengths: list[int] = [len(group) for group in data]
        self._padding: list[int] = [
            ceil(size / round_to_multiple) * round_to_multiple - size
            for size in self._group_lengths
        ]

    def __iter__(self) -> Iterator[T | None]:
        """
        Provides the flattened data stream: Iterable[Iterable[T]] -> Iterable[T].
        Returns an iterator over the cached data.
        """
        for i, group in enumerate(self._data):
            yield from group
            yield from [None] * self._padding[i]

    def indices(self) -> Iterator[tuple[int, int] | None]:
        """
        Provides the (set index, index within set) mapping.
        This is always correct as the structure is pre-computed.
        """
        for i, length in enumerate(self._group_lengths):
            for j in range(length):
                yield (i, j)
            yield from [None] * self._padding[i]

    def set_indices(self) -> Iterator[int | None]:
        yield from (idx[0] if idx else None for idx in self.indices())
