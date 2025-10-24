from collections.abc import Iterable, Iterator
from typing import cast, final, override


@final
class Zipper[T, U](Iterable[tuple[T, U]]):
    """
    streaming [...T], [...U] -> [... (T,U) ]
    """

    def __init__(self, iter1: Iterable[T], iter2: Iterable[U]):
        self.iterables = [iter1, iter2]

    @override
    def __iter__(self) -> Iterator[tuple[T, U]]:
        iterators = [iter(it) for it in self.iterables]

        if not iterators:
            return

        while True:
            try:
                item = tuple([next(it) for it in iterators])
                yield cast(tuple[T, U], item)
            except StopIteration:
                return
