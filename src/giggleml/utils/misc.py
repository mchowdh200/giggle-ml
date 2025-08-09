from typing import cast

from torch import Tensor


def partition_integer(n: int, k: int) -> list[int]:
    """
    Splits an integer n into k parts as evenly as possible.

    Args:
      n: The integer to split.
      k: The number of parts.

    Returns:
      A list of k integers that sum to n.
    """
    if k <= 0:
        return []

    base_size = n // k
    remainder = n % k

    parts = []
    for i in range(k):
        if i < remainder:
            parts.append(base_size + 1)
        else:
            parts.append(base_size)

    return parts


def partition_list[T: list | Tensor](items: T, world_size: int, rank: int) -> T:
    """
    Splits a list into partitions, as evenly possible, and returns the
    parition corresponding to the rank.
    """
    splits = partition_integer(len(items), world_size)
    i = sum(splits[:rank])
    part = splits[rank]
    return cast(T, items[i : i + part])
