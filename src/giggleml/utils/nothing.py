from typing import Iterable, Iterator, overload


# it does nothing
class Nothing:
    """A callable that does nothing but return its first argument unchanged or None."""

    @overload
    def __call__(self) -> None: ...

    @overload
    def __call__[T](self, arg: T) -> T: ...

    @overload
    def __call__[T](self, arg: T, *args: object) -> T: ...

    def __call__[T](self, arg: T | None = None, *args: object) -> T | None:
        if len(args) != 0:
            # what do you expect to do, return multiple args?
            raise RuntimeError("Nothing can only be called with zero or one argument")
        return arg

    def __iter__(self):
        """Generator that yields nothing."""
        return
        yield  # unreachable, but makes this a generator


class YieldFrom[T]:
    """A callable that yields all items from an iterable unchanged."""

    def __call__(self, iterable: Iterable[T]) -> Iterator[T]:
        yield from iterable


class YieldThrough[T]:
    """A callable that yields the item unchanged."""

    def __call__(self, thing: T) -> Iterator[T]:
        yield thing


nothing = Nothing()
yield_through = YieldThrough()
yield_from = YieldFrom()
