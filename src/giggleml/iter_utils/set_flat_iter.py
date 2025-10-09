from collections.abc import Iterable, Iterator


class SetFlatIter[T, U]:
    """
    A streaming-friendly utility to flatten a nested iterable and later
    regroup a corresponding flat iterable back into the original structure.
    """

    def __init__(self, nested_iterable: Iterable[Iterable[T]]):
        """
        Initializes the iterator without consuming the input.

        Args:
            nested_iterable: An iterable of iterables (e.g., a generator of lists).
        """
        self._nested_iterable = nested_iterable
        self._group_lengths: list[int] = []
        # A flag to ensure the iterator is consumed only once.
        self._iteration_started = False

    def __iter__(self) -> Iterator[T]:
        """
        Provides the flattening functionality: Iterable[Iterable[T]] -> Iterable[T].

        As items are yielded, the structure of the groups is recorded internally.
        This iterator can only be fully consumed once.
        """
        if self._iteration_started:
            raise RuntimeError(
                "This streaming iterator has already been consumed or started."
            )
        self._iteration_started = True

        # This generator function is the core of the flattening process
        for group in self._nested_iterable:
            count = 0
            for item in group:
                yield item
                count += 1
            # After each inner group is exhausted, we now know its length.
            self._group_lengths.append(count)

    def set_indices(self) -> list[int]:
        "Provides the set indices mapping for the flattened structure."

        if not self._group_lengths:
            raise RuntimeError(
                "Must fully iterate over the object before calling regroup(). "
                "The group structure is not yet known."
            )

        def iterate():
            for i, count in enumerate(self._group_lengths):
                yield from [i] * count

        return list(iterate())

    def regroup(self, flat_iterable: Iterable[U]) -> Iterator[list[U]]:
        """
        Provides the inverse functionality: Iterable[U] -> Iterable[Iterable[U]].

        This method MUST be called after the main iterator has been fully consumed.
        It uses the recorded group lengths to structure the new flat iterable.

        Args:
            flat_iterable: An iterable of items to be regrouped.

        Returns:
            An iterator that yields lists of regrouped items.
        """

        if not self._group_lengths:
            raise RuntimeError(
                "Must fully iterate over the object before calling regroup(). "
                "The group structure is not yet known."
            )

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
