from typing import overload


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
        return arg

    def __iter__(self):
        """Generator that yields nothing."""
        return
        yield  # unreachable, but makes this a generator

nothing = Nothing()
