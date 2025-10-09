import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import cast, overload

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
            Elements assigned to current rank
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Each worker processes every world_size-th element starting from rank
            yield from itertools.islice(self.data, rank, None, world_size)
        else:
            yield from self.data

    @overload
    @staticmethod
    def inverse(splits: Iterable[torch.Tensor]) -> torch.Tensor: ...

    @overload
    @staticmethod
    def inverse(splits: Iterable[Iterable[T]]) -> list[T]: ...

    @staticmethod
    def inverse(
        splits: Iterable[Iterable[T] | torch.Tensor],
    ) -> list[T] | torch.Tensor:
        """
        Inverts the rank splitting, recombining into a single iterable or tensor.
        Takes the total set of iterables across all ranks and interleaves them.

        If contains Tensors, it efficiently interleaves their rows.
        Otherwise, it interleaves the elements of the iterables.
        """

        # Eagerly convert to a list to check the type and get the length.
        # This is necessary for both logic paths.
        splits_list: Sequence[torch.Tensor | Iterable[T]] = list(splits)

        if not splits_list:
            # Handle empty input gracefully for both cases
            if isinstance(splits, Iterable) and not isinstance(
                next(iter(splits), None), torch.Tensor
            ):
                # Return an empty iterator for the generic case
                return iter(list())  # pyright: ignore[reportReturnType]
            return torch.tensor([])  # Return an empty tensor otherwise

        # --- Tensor Path ---
        if isinstance(splits_list[0], torch.Tensor):
            splits_list = [cast(torch.Tensor, s) for s in splits_list]

            # Check if all tensors have the same shape (the common, fast case)
            first_shape = splits_list[0].shape
            is_even = all(s.shape == first_shape for s in splits_list[1:])

            if is_even:
                # 1. Stack along a new dimension. Shape: (world_size, n_rows, *features)
                # 2. Transpose the first two dims. Shape: (n_rows, world_size, *features)
                #    This groups the rows from each rank together.
                # 3. Reshape to flatten the first two dims. Shape: (n_rows * world_size, *features)
                stacked_tensors = torch.stack(splits_list, dim=0)
                transposed = stacked_tensors.transpose(0, 1)
                return transposed.reshape(-1, *first_shape[1:])

            else:
                max_rows = max(s.size(0) for s in splits_list)
                interleaved_rows = (
                    s[i] for i in range(max_rows) for s in splits_list if i < s.size(0)
                )
                return torch.stack(list(interleaved_rows), dim=0)

        # --- Pure Iterables Path ---
        sentinel = object()  # Unique object to detect the end of shorter iterables
        zipped = itertools.zip_longest(*splits_list, fillvalue=sentinel)
        chained: Iterator[T | object] = iter(itertools.chain.from_iterable(zipped))
        # Filter out the sentinel values for iterables that finished early
        return cast(list[T], list(filter(lambda x: x is not sentinel, chained)))
