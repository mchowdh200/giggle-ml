import itertools
from collections.abc import Iterable, Iterator
from typing import overload

import torch
import torch.distributed as dist

from giggleml.utils.torch_utils import rprint


class RankIter[T]:
    """
    Higher-order iterable that correctly shards data for distributed processing,
    even when used with a DataLoader (num_workers > 0).

    It captures the distributed rank/world_size when instantiated and
    combines it with the DataLoader worker_id when iterated.
    """

    def __init__(self):
        """
        Initialize with source data iterable and capture distributed context.

        This __init__ method is expected to be called from the main
        distributed process (not a DataLoader worker).

        Args:
            data: Source iterable to shard across ranks and workers
        """

        # 1. Capture rank and world size from the main process
        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1

        rprint(f"RankIter.__init__: dist.is_initialized()={dist.is_initialized()}")

    def iter(self, data: Iterable[T]) -> Iterator[T]:
        """
        Iterate through elements, sharded by rank and/or worker.

        This __iter__ method is expected to be called by the process
        that consumes the data (e.g., a DataLoader worker).

        Yields:
            Elements assigned to the current rank and worker
        """

        # 2. Get worker info from the worker process (or None if num_workers=0)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            rprint(
                f"RankIter.__iter__: worker_info={worker_info}, dist.is_initialized()={dist.is_initialized()}"
            )

        data_iter = iter(data)

        if worker_info is None:
            # We are in the main process (num_workers = 0)
            # Shard by rank only
            start_index = self.rank
            step_size = self.world_size
        else:
            # We are in a worker process (num_workers > 0)
            # Shard by both rank and worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # The global index for this specific worker on this specific rank
            start_index = self.rank * num_workers + worker_id

            # The total number of workers across all ranks
            step_size = self.world_size * num_workers

        # 3. Yield the correctly sharded data
        yield from itertools.islice(data_iter, start_index, None, step_size)

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
