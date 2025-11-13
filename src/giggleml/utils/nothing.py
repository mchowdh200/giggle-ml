from collections.abc import Iterable, Iterator
from typing import Any, overload


# it does nothing
class Nothing:
    """A callable that does nothing but return its first argument unchanged or None."""

    @overload
    def __call__(self) -> None: ...

    @overload
    def __call__[T](self, arg: T, *args: Any, **kwargs: Any) -> T: ...

    @overload
    def __call__[T](self, arg: None = None, *args: Any, **kwargs: Any) -> None: ...

    def __call__[T](self, arg: T | None = None, *args: Any, **kwargs: Any) -> T | None:
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
