import functools
from collections import OrderedDict
from typing import Callable, overload


class LRUCache[K, T]:
    """
    not thread safe
    """

    def __init__(self, limit: int = 128):
        self._limit: int = limit
        self._cache: OrderedDict[K, T] = OrderedDict()

    def get(self, key: K, default: Callable[[], T]) -> T:
        if key in self._cache:
            # Move the accessed key to the end (making it the most recent)
            self._cache.move_to_end(key)
            return self._cache[key]

        # If not in cache, call the factory function to get the value
        value = default()
        self._cache[key] = value

        # If the cache is over the limit, pop the first item (least recent)
        if len(self._cache) > self._limit:
            self._cache.popitem(last=False)

        return value

    def pop(self, key: K) -> T:
        if key not in self._cache:
            raise RuntimeError("missing key")
        return self._cache.pop(key)

    def add(self, key: K, value: T):
        """Adds or updates an item in the cache."""
        self._cache[key] = value
        self._cache.move_to_end(key)  # Mark as most recent

        if len(self._cache) > self._limit:
            self._cache.popitem(last=False)


@overload
def lru_cache[**P, T](
    *, max_size: int = 8192, typed: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


# Overload for when the decorator is called without arguments, e.g., @lru_cache
@overload
def lru_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]: ...


# Implementation of the decorator
def lru_cache[**P, T](
    func: Callable[P, T] | None = None, *, max_size: int = 8192, typed: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
    """
    A strongly-typed wrapper for functools.lru_cache

    Usage:
        @lru_cache
        def my_function(...): ...

        @lru_cache(limit=128, typed=True)
        def other_function(...): ...
    """

    # The inner decorator is also generic. It captures the type variables P and T
    # from the outer scope and applies them to its signature.
    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        # Note: The underlying functools.lru_cache uses 'maxsize'.
        return functools.lru_cache(maxsize=max_size, typed=typed)(f)  # pyright: ignore[reportReturnType]

    if func is not None:
        return decorator(func)
    else:
        # Case 2: Called as @lru_cache(limit=...) (with parentheses)
        return decorator
