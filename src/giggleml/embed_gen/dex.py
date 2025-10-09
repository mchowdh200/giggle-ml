"""
Dex: Distributed Executor

An optimized, distributed model execution utility designed for flexible composition
across various use cases. Provides streaming processing from Iterable[T] -> Iterable[Tensor].

Pipeline Flow:
    input element -> [pre-processor] -> [DataLoader, collate] -> model -> [post-processor] -> output element

Key Design Principles:
- Pre-processor runs before DataLoader/collate to allow element multiplication
- DataLoader defines batch boundaries after pre-processing
- Pre/post processors can add or remove elements:
  * Preprocessor: T -> Iterable[U] (one-to-many)
  * Postprocessor: Iterable[U] -> T (many-to-one)
- Environment-agnostic: uses Consumer interface instead of direct returns
- Supports distributed execution with PyTorch distributed

The collate function is called after the batch boundaries have been set, preprocess
can be used to extend the dataset because its called prior to fixed batch boundaries.
"""

import itertools
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from giggleml.iter_utils.rank_iter import RankIter
from giggleml.utils.torch_utils import guess_device

# Type aliases for pipeline components
type PreprocessorFn[T_in, U_pre] = Callable[[T_in], Iterable[U_pre]]
"""Transforms input elements, potentially creating multiple outputs per input."""

type PostprocessorFn[V_post, W_out] = Callable[[Iterable[V_post]], W_out]
"""Aggregates multiple model outputs back into a single result."""

type ConsumerFn[W_out] = Callable[[W_out], None]
"""Handles final outputs (e.g., caching, writing to files)."""

type CollateFn[U_pre, Batch_in] = Callable[
    [list[tuple[U_pre, bool]]], tuple[Batch_in, list[bool]]
]
"""Combines preprocessed items into batches with boundary flags."""

type DecollateFn[Batch_out, V_post] = Callable[
    [Batch_out, list[bool]], Iterable[tuple[V_post, bool]]
]
"""Splits model batch outputs back into individual items with boundary flags."""


class _StreamingPreprocessorDataset[T_in, U_pre](IterableDataset):
    """Dataset that applies preprocessing and handles distributed data sharding.

    Automatically distributes data across workers in distributed settings and
    tracks element boundaries for proper postprocessing.
    """

    def __init__(
        self, data: Iterable[T_in], preprocessor_fn: PreprocessorFn[T_in, U_pre]
    ):
        """Initialize dataset with data source and preprocessing function.

        Args:
            data: Input data iterable
            preprocessor_fn: Function to transform each input element
        """
        super().__init__()
        self.data = data
        self.preprocessor_fn = preprocessor_fn

    def __iter__(self) -> Iterator[tuple[U_pre, bool]]:
        """Iterate through preprocessed data with boundary flags.

        Uses RankIter for distributed data sharding. Returns tuples of
        (preprocessed_item, is_last_in_group) to track element boundaries
        for postprocessing.

        Yields:
            Tuples of (preprocessed_item, is_last_flag)
        """

        # distributed data sharding
        rank_data = RankIter(self.data)

        for item in rank_data:
            # Apply preprocessing - may produce multiple outputs per input
            sub_items_iter = iter(self.preprocessor_fn(item))

            # Stream through sub-items, marking the last one
            null_item = object()  # sentinel
            prev_item: U_pre | object = null_item

            for sub_item in sub_items_iter:
                if prev_item is not null_item:
                    # Yield previous item (not last)
                    yield (prev_item, False)  # pyright: ignore[reportReturnType]

                prev_item = sub_item

            if prev_item is not null_item:
                yield (prev_item, True)  # pyright: ignore[reportReturnType]


class Dex[T_in, U_pre, V_post, W_out, Batch_in, Batch_out]:
    """
    Dex: Distributed Executor

    A flexible, streaming model execution pipeline supporting distributed processing.

    Type Parameters:
        T_in: Input data type
        U_pre: Preprocessed data type
        V_post: Model output type (per item)
        W_out: Final output type (after postprocessing)
        Batch_in: Batched input type for model
        Batch_out: Batched output type from model

    The pipeline flow:
        1. Preprocess: T_in -> Iterable[U_pre]
        2. Collate: List[U_pre] -> Batch_in
        3. Model: Batch_in -> Batch_out
        4. Decollate: Batch_out -> Iterable[V_post]
        5. Postprocess: Iterable[V_post] -> W_out
        6. Consume: W_out -> None
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor_fn: PreprocessorFn[T_in, U_pre],
        postprocessor_fn: PostprocessorFn[V_post, W_out],
        collate_fn: CollateFn[U_pre, Batch_in],
        decollate_fn: DecollateFn[Batch_out, V_post],
    ):
        """Initialize Dex with model and pipeline functions.

        Args:
            model: PyTorch model to execute
            preprocessor_fn: Transforms input data (T_in -> Iterable[U_pre])
            postprocessor_fn: Aggregates outputs (Iterable[V_post] -> W_out)
            collate_fn: Batches preprocessed data for model
            decollate_fn: Unbatches model outputs
        """
        self.model: torch.nn.Module = model
        self.preprocessor_fn: PreprocessorFn[T_in, U_pre] = preprocessor_fn
        self.postprocessor_fn: PostprocessorFn[V_post, W_out] = postprocessor_fn
        self.collate_fn: CollateFn[U_pre, Batch_in] = collate_fn
        self.decollate_fn: DecollateFn[Batch_out, V_post] = decollate_fn

    def execute(
        self,
        data: Iterable[T_in],
        consumer_fn: ConsumerFn[W_out],
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        """Execute the full pipeline on input data.

        Args:
            data: Input data iterable
            consumer_fn: Function to handle final outputs
            batch_size: Number of items per batch
            num_workers: Number of DataLoader workers
        """
        # Check if we're in a distributed environment and wrap model with DDP
        model = self.model
        current_rank = 0
        
        if dist.is_available() and dist.is_initialized():
            current_rank = dist.get_rank()
        
        # Automatically infer device based on current rank
        device = guess_device(current_rank)
        
        if dist.is_available() and dist.is_initialized():
            if not isinstance(model, DDP):
                model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
        
        # Create dataset that handles preprocessing and distributed sharding
        dataset = _StreamingPreprocessorDataset(data, self.preprocessor_fn)

        # Setup DataLoader with custom collation
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",  # Optimize for GPU transfers
        )

        def _model_pass(
            loader: DataLoader,
        ) -> Iterator[tuple[Batch_out, list[bool]]]:
            """Execute model on batched data.

            Handles device transfer for tensor components and preserves
            boundary flags for proper postprocessing.
            """
            for batch_input, boundary_flags in loader:
                # Move only tensor components to target device
                moved_batch_components = tuple(
                    t.to(device) for t in batch_input if isinstance(t, torch.Tensor)
                )
                # Reconstruct batch with moved tensors
                full_batch_for_model = (batch_input[0], *moved_batch_components)

                # Execute model
                output = model(full_batch_for_model)
                yield output, boundary_flags

        # Execute model on all batches
        model_output_batches = _model_pass(data_loader)

        # Flatten batched outputs back to item stream
        output_item_stream = itertools.chain.from_iterable(
            self.decollate_fn(batch_output, flags)
            for batch_output, flags in model_output_batches
        )

        # Group items and apply postprocessing
        final_results = self._regroup_and_postprocess(output_item_stream)

        # Send results to consumer
        for result in final_results:
            consumer_fn(result)

    def _regroup_and_postprocess(
        self, items: Iterable[tuple[V_post, bool]]
    ) -> Iterator[W_out]:
        """Regroup items by boundary flags and apply postprocessing.

        The boundary flags from preprocessing are used to group related items
        back together before applying the postprocessor function.

        Args:
            items: Stream of (item, is_last_in_group) tuples

        Yields:
            Postprocessed results
        """
        current_group: list[V_post] = []
        for item, is_last in items:
            current_group.append(item)
            if is_last:
                # End of group - apply postprocessing
                yield self.postprocessor_fn(current_group)
                current_group = []

        # Handle any remaining items in the final group
        if current_group:
            yield self.postprocessor_fn(current_group)
