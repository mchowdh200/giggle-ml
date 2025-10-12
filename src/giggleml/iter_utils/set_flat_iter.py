import itertools
from collections.abc import Collection, Iterable, Iterator


class SetFlatIter[T, U]:
    """
    A utility to flatten a nested iterable and later regroup a
    corresponding flat iterable back into the original structure.

    This implementation performs a single pass on initialization to learn the
    full structure, caching the data to ensure correctness and consistency
    between its methods.
    """

    def __init__(self, data: Collection[Collection[T]]):
        """
        Initializes and consumes the input to learn the structure and cache data.
        """
        self._data: Collection[Collection[T]] = data
        self._group_lengths: list[int] = [len(group) for group in data]

    def __iter__(self) -> Iterator[T]:
        """
        Provides the flattened data stream: Iterable[Iterable[T]] -> Iterable[T].
        Returns an iterator over the cached data.
        """
        yield from itertools.chain.from_iterable(self._data)

    def indices(self) -> Iterator[tuple[int, int]]:
        """
        Provides the (set index, index within set) mapping.
        This is always correct as the structure is pre-computed.
        """
        for i, length in enumerate(self._group_lengths):
            for j in range(length):
                yield (i, j)

    def set_indices(self) -> Iterator[int]:
        yield from (i for (i, _) in self.indices())

    def regroup(self, flat_iterable: Iterable[U]) -> Iterator[list[U]]:
        """
        Regroups a flat iterable using the pre-computed structure.
        """
        flat_iter = iter(flat_iterable)

        for length in self._group_lengths:
            try:
                yield [next(flat_iter) for _ in range(length)]
            except StopIteration:
                raise ValueError(
                    "The 'flat_iterable' has fewer items than the original structure."
                )

        # This check ensures the input iterable's length matches the original
        try:
            next(flat_iter)
            raise ValueError(
                "The 'flat_iterable' has more items than the original structure."
            )
        except StopIteration:
            pass
