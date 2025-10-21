import itertools
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset

# Assumes these local utility imports exist
from giggleml.iter_utils.rank_iter import RankIter
from giggleml.utils.nothing import nothing, yield_through
from giggleml.utils.torch_utils import get_world_size, guess_device, is_distributed

# Type aliases for pipeline components
type PreprocessorFn[T_in, U_pre] = Callable[[T_in], Iterable[U_pre]]
"""Transforms input elements, potentially creating multiple outputs per input."""

type PostprocessorFn[V_post, W_out] = Callable[[Iterable[V_post]], W_out]
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

    def __iter__(self) -> Iterator[tuple[U_pre, bool]]:
        """Iterate through preprocessed data with correct boundary flags."""
        blocks = itertools.batched(self.data, self.batch_size)
        rank_data = RankIter(blocks)

        for block in rank_data:
            # Process each item in the block individually to create
            # distinct groups for the postprocessor.
            for item in block:
                sub_items_iter = iter(self.preprocessor_fn(item))

                try:
                    prev_sub_item = next(sub_items_iter)
                except StopIteration:
                    continue  # Preprocessor produced no items for this input

                for sub_item in sub_items_iter:
                    yield (prev_sub_item, False)
                    prev_sub_item = sub_item

                # Mark the end of the group for this specific 'item'.
                yield (prev_sub_item, True)


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

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor_fn: PreprocessorFn[T_in, U_pre] = yield_through,
        postprocessor_fn: PostprocessorFn[V_post, W_out] = nothing,
        collate_fn: CollateFn[U_pre, Batch_in] = nothing,
        decollate_fn: DecollateFn[Batch_out, V_post] = yield_through,
    ):
        """Initialize Dex with model and pipeline functions."""
        self.model: torch.nn.Module = model
        self.preprocessor_fn = preprocessor_fn
        self.postprocessor_fn = postprocessor_fn
        self.collate_fn = collate_fn
        self.decollate_fn = decollate_fn

    def _internal_collate(
        self, batch: list[tuple[U_pre, bool]]
    ) -> tuple[Batch_in, list[bool]]:
        """Internal collator to separate data from flags before calling user collate_fn."""
        items = [item for item, _ in batch]
        flags = [flag for _, flag in batch]
        # Call the user-provided collate function with only the data
        user_batch = self.collate_fn(items)
        return user_batch, flags

    def simulate[T](self, data: Iterable[T], batch_size: int) -> Iterator[Iterable[T]]:
        """Pass data through, without executing the pipeline, for indices tracking"""
        blocks = itertools.batched(data, batch_size)
        rank_data = RankIter(blocks)
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
            batch_size=batch_size,
            collate_fn=self._internal_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        # 1. Execute model on all batches
        model_output_batches = self._model_pass(data_loader, model, device)

        # 2. Flatten batched outputs back to an item stream with boundary flags
        output_item_stream = self._decollate_and_rebatch(model_output_batches)

        # 3. Group items by boundary flags and apply postprocessing
        final_results = self._regroup_and_postprocess(output_item_stream)

        # 4. Send results to the consumer
        for block in itertools.batched(final_results, batch_size):
            consumer_fn(block)

    def _model_pass(
        self, loader: DataLoader, model: torch.nn.Module, device: torch.device
    ) -> Iterator[tuple[Batch_out, list[bool]]]:
        """Executes the model on batched data, handling device placement."""
        for batch_input, boundary_flags in loader:
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
            yield output, boundary_flags

    def _decollate_and_rebatch(
        self, model_output_batches: Iterable[tuple[Batch_out, list[bool]]]
    ) -> Iterator[tuple[V_post, bool]]:
        """Applies user decollate and re-associates items with boundary flags."""
        for batch_output, flags in model_output_batches:
            # User decollate function returns an iterable of items
            decollated_items = self.decollate_fn(batch_output)
            # Zip items with their corresponding flags
            yield from zip(decollated_items, flags)

    def _regroup_and_postprocess(
        self, items: Iterable[tuple[V_post, bool]]
    ) -> Iterator[W_out]:
        """Regroups items by boundary flags and applies postprocessing."""
        current_group: list[V_post] = []

        for item, is_last in items:
            current_group.append(item)

            if is_last:
                yield self.postprocessor_fn(current_group)
                current_group = []

        # Handle any remaining items if the stream ends mid-group
        if current_group:
            yield self.postprocessor_fn(current_group)
