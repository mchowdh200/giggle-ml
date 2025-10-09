import itertools
from collections.abc import Iterable, Iterator
from typing import overload

import torch
import torch.distributed as dist


class RankIter[T](Iterable[T]):
    """Higher-order iterable that yields only elements for the current distributed rank.

    In distributed settings, automatically shards data across workers by yielding
    every world_size-th element starting from the current rank. In non-distributed
    settings, yields all elements.
    """

    def __init__(self, data: Iterable[T]):
        """Initialize with source data iterable.

        Args:
            data: Source iterable to shard across ranks
        """
        self.data = data

    def __iter__(self) -> Iterator[T]:
        """Iterate through rank-appropriate elements.

        Yields:
            Elements assigned to the current rank
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Each worker processes every world_size-th element starting from its rank
            yield from itertools.islice(self.data, rank, None, world_size)
        else:
            yield from self.data

    @overload
    @staticmethod
    def inverse(splits: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]: ...

    @overload
    @staticmethod
    def inverse(splits: Iterable[Iterable[T]]) -> Iterator[T]: ...

    @staticmethod
    def inverse(
        splits: Iterable[Iterable[T] | torch.Tensor],
    ) -> Iterator[T | torch.Tensor]:
        """Inverts rank splitting, recombining splits into a single lazy iterator.

        This method takes the separated iterables from all ranks and interleaves
        their elements in a streaming fashion to reconstruct the original order.

        If the iterables contain Tensors, it efficiently interleaves their rows.
        Otherwise, it interleaves the elements of the generic iterables.

        Args:
            splits: An iterable containing the distributed data splits from each rank.

        Yields:
            Elements from the splits, interleaved in their original order.
        """
        # Eagerly convert the outer container of splits. This is a reasonable
        # compromise as the number of splits (world_size) is typically small.
        # The inner iterables, which contain the actual data, will be streamed.
        splits_list = list(splits)
        if not splits_list:
            return  # A generator function returns an empty iterator by default

        # The logic is identical for generic iterables and for Tensors, where
        # a Tensor is treated as an iterable of its rows.
        sentinel = object()  # Unique object to detect the end of shorter iterables

        # 1. `zip_longest` gets the i-th element (or row) from each split in lockstep.
        #    e.g., (rank0_item0, rank1_item0), (rank0_item1, rank1_item1), ...
        interleaved_groups = itertools.zip_longest(*splits_list, fillvalue=sentinel)

        # 2. `chain.from_iterable` flattens these groups into a single stream.
        #    e.g., rank0_item0, rank1_item0, rank0_item1, rank1_item1, ...
        flat_stream = itertools.chain.from_iterable(interleaved_groups)

        # 3. `filter` removes the sentinel padding from shorter iterables and yields.
        #    The entire pipeline is lazy and processes one item at a time.
        yield from filter(lambda x: x is not sentinel, flat_stream)  # pyright: ignore[reportReturnType]
