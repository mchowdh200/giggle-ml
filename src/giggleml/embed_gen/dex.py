import itertools
from collections.abc import Callable, Iterable, Iterator
from logging import warning

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    default_collate,
    get_worker_info,
)

# Assumes these local utility imports exist
from giggleml.iter_utils.rank_iter import RankIter
from giggleml.utils.nothing import nothing, yield_from, yield_through
from giggleml.utils.torch_utils import (
    get_world_size,
    guess_device,
    is_distributed,
)

# Type aliases for pipeline components
type PreprocessorFn[T_in, U_pre] = Callable[[T_in], Iterable[U_pre]]
"""Transforms input elements, potentially creating multiple outputs per input."""

type PostprocessorFn[V_post, W_out] = Callable[[list[V_post]], W_out]
"""Aggregates multiple model outputs back into a single result."""

type ConsumerFn[W_out] = Callable[[Iterable[W_out]], None]
"""Handles final outputs (e.g., caching, writing to files)."""

type CollateFn[U_pre, Batch_in] = Callable[[list[U_pre]], Batch_in]
"""Combines preprocessed items into a batch for the model."""

type DecollateFn[Batch_out, V_post] = Callable[[Batch_out], Iterable[V_post]]
"""Splits model batch outputs back into individual items."""


class _StreamingPreprocessorDataset[T_in, U_pre](IterableDataset):
    """Dataset that applies preprocessing and handles distributed data sharding."""

    def __init__(
        self,
        data: Iterable[T_in],
        batch_size: int,
        preprocessor_fn: PreprocessorFn[T_in, U_pre],
    ):
        """Initialize dataset with data source and preprocessing function."""
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.preprocessor_fn = preprocessor_fn

        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1

    def __iter__(self) -> Iterator[tuple[int, U_pre]]:
        """Iterate through preprocessed data with regroup indices."""
        # batching at the top level prevents consumers from getting mixed-chunk items
        total_blocks = itertools.batched(self.data, self.batch_size)
        # a flat stream from the blocks for this rank
        rank_items = itertools.chain.from_iterable(
            itertools.islice(total_blocks, self.rank, None, self.world_size)
        )
        worker_info = get_worker_info()
        sub_rank = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        # de-interleave of the rank's share into sub-worker tasks
        sub_worker_items = itertools.islice(rank_items, sub_rank, None, num_workers)

        for item in sub_worker_items:
            # we make the required assumption that the expanded_item group fits in memory
            expanded_items = list(self.preprocessor_fn(item))
            yield from ((sub_rank + 1, item) for item in expanded_items[:-1])
            yield (
                -(sub_rank + 1),
                expanded_items[-1],
            )  # mark last item with negative index
            # we add one because zero can't be signed


class Dex[T_in, U_pre, V_post, W_out, Batch_in, Batch_out]:
    """
    Dex: Distributed Executor

    Distributed model execution utility designed for flexible composition across
    various use cases. Provides streaming processing from Iterable[T] -> Iterable[Tensor].

    Pipeline Flow:
        input element -> [pre-processor] -> [DataLoader, collate] -> [model] -> [post-processor] -> output element

    Iterates the data in blocks to allow subsequent systems to write outputs in blocks, such as zarr chunks.

    Key Design Principles:
    - Pre-processor runs before DataLoader/collate to allow element multiplication.
    - DataLoader defines batch boundaries after pre-processing. The user's collate
      function is called after these boundaries are set, whereas the pre-processor is
      called before, allowing it to expand one input into many processed items.
    - Pre/post processors can add or remove elements:
      * Preprocessor: T -> Iterable[U] (one-to-many)
      * Postprocessor: Iterable[U] -> T (many-to-one)
    - Environment-agnostic: uses Consumer interface instead of direct returns.
    - Supports distributed execution with PyTorch distributed.

    @param collate_fn must be picklable for DataLoader compatibility.
    """

    class _InternalCollate:
        """Internal collator to separate data from flags before calling user collate_fn."""

        def __init__(self, base_collate_fn: CollateFn) -> None:
            self.collate_fn: CollateFn = base_collate_fn

        def __call__(
            self, batch: list[tuple[int, U_pre]]
        ) -> tuple[list[int], Batch_in]:
            indices, items = zip(*batch)
            items = list(items)
            indices = list(indices)
            # Call the user-provided collate function with only the data
            user_batch = self.collate_fn(items)
            return indices, user_batch

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor_fn: PreprocessorFn[T_in, U_pre] = yield_through,
        postprocessor_fn: PostprocessorFn[V_post, W_out] = nothing,
        collate_fn: CollateFn[U_pre, Batch_in] = default_collate,
        decollate_fn: DecollateFn[Batch_out, V_post] = yield_from,
    ):
        """Initialize Dex with model and pipeline functions."""
        self.model: torch.nn.Module = model
        self.preprocessor_fn = preprocessor_fn
        self.postprocessor_fn = postprocessor_fn
        self.collate_fn = collate_fn
        self.decollate_fn = decollate_fn

    def simulate[T](self, data: Iterable[T], batch_size: int) -> Iterator[Iterable[T]]:
        """Pass data through, without executing the pipeline, for indices tracking"""
        blocks = itertools.batched(data, batch_size)
        rank_data = RankIter().iter(blocks)
        yield from rank_data

    def simulate_global_concat[T](self, data: Iterable[T], batch_size: int) -> list[T]:
        """
        Simulate distributed batching and concatenate all ranks' data in order.
        Outputs for each rank are individually concatenated, then those concatenated.
        """
        world_size = get_world_size()
        ranks = [list() for _ in range(world_size)]

        for i, block in enumerate(itertools.batched(data, batch_size)):
            rank_idx = i % world_size
            ranks[rank_idx].extend(block)

        return sum(ranks, list())

    def execute(
        self,
        data: Iterable[T_in],
        consumer_fn: ConsumerFn[W_out],
        batch_size: int,
        num_workers: int = 0,
    ):
        """Execute the full pipeline on input data."""
        # Setup device and distributed model if applicable
        model = self.model
        current_rank = dist.get_rank() if is_distributed() else 0
        device = guess_device(current_rank)
        model.to(device)

        if is_distributed():
            mp.set_start_method("spawn", force=True)

        # Create dataset that handles preprocessing and distributed sharding
        dataset = _StreamingPreprocessorDataset(data, batch_size, self.preprocessor_fn)

        # Setup DataLoader with the internal (picklable) collation method
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self._InternalCollate(self.collate_fn),
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            pin_memory=device.type == "cuda",
        )

        # 1. Execute model on all batches
        model_output_batches = self._model_pass(data_loader, model, device)

        # 2. Flatten batched outputs back to an item stream with boundary flags
        output_item_stream = self._decollate_and_rebatch(model_output_batches)

        # 3. Group items by boundary flags and apply postprocessing
        final_results = self._regroup_and_postprocess(num_workers, output_item_stream)

        # 4. Send results to the consumer
        for block in itertools.batched(final_results, batch_size):
            consumer_fn(block)

    def _model_pass(
        self, loader: DataLoader, model: torch.nn.Module, device: torch.device
    ) -> Iterator[tuple[list[int], Batch_out]]:
        """Executes the model on batched data, handling device placement."""
        for group_indices, batch_input in loader:
            # Robustly move batch components to the target device
            if isinstance(batch_input, (list, tuple)):
                moved_batch = tuple(
                    item.to(device, non_blocking=True)
                    if isinstance(item, torch.Tensor)
                    else item
                    for item in batch_input
                )
            elif isinstance(batch_input, torch.Tensor):
                moved_batch = batch_input.to(device, non_blocking=True)
            else:
                moved_batch = batch_input  # Assume it's a custom object

            output = model(moved_batch)
            yield group_indices, output

    def _decollate_and_rebatch(
        self, model_output_batches: Iterable[tuple[list[int], Batch_out]]
    ) -> Iterator[tuple[int, V_post]]:
        """Applies user decollate and re-associates items with boundary flags."""
        for group_indices, batch_output in model_output_batches:
            # User decollate function returns an iterable of items
            decollated_items = self.decollate_fn(batch_output)
            # Zip items with their corresponding flags
            yield from zip(group_indices, decollated_items)

    def _regroup_and_postprocess(
        self, num_workers: int, items: Iterable[tuple[int, V_post]]
    ) -> Iterator[W_out]:
        """Regroups items by group index and applies postprocessing."""
        groups: list[list[V_post]] = [list() for _ in range(num_workers)]

        for signed_group_idx, item in items:
            is_last = signed_group_idx < 0
            group_idx = abs(signed_group_idx) - 1

            current_group = groups[group_idx]
            current_group.append(item)

            if is_last:
                yield self.postprocessor_fn(current_group)
                current_group.clear()

        # Handle any remaining items if the stream ends mid-group
        for group in groups:
            if len(group) != 0:
                # this should not happen
                warning(
                    "Output stream ended before group was closed; flushing remaining items."
                )
                yield self.postprocessor_fn(group)
