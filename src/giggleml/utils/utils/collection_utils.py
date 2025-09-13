import functools
from collections.abc import Callable, Iterable


def take_key[K, V](key: K, items: Iterable[dict[K, V]]) -> Iterable[V]:
    """maps each dict to the specified key"""

    for item in items:
        yield item[key]


def as_list[**P, T](func: Callable[P, Iterable[T]]) -> Callable[P, list[T]]:
    """
    A decorator that wraps a generator function and returns its output as a list.
    This forces the immediate and complete evaluation of the generator.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> list[T]:
        return list(func(*args, **kwargs))

    return wrapper