from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import final

from torch import Tensor

from giggleml.embed_gen.multi_zarr_writer import MultiZarrWriter
from giggleml.iter_utils.set_flat_iter import SetFlatIter
from giggleml.utils.types import lazy

from ..data_wrangling.interval_dataset import IntervalDataset
from .dex import ConsumerFn, Dex
from .embed_model import EmbedModel
from .genomic_dex import GenomicDex


@lazy
@final
class BatchInfer:
    def __init__(
        self,
        model: EmbedModel,
        batch_size: int,
        num_workers: int = 0,
    ):
        """
        @param num_workers: Number of DataLoader workers for batch construction
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embed_dim = model.embed_dim

    def raw(
        self,
        datasets: Sequence[IntervalDataset],
        consumer: ConsumerFn[Tensor],
    ) -> None:
        """Process datasets with a generic consumer function.

        The consumer receives individual tensors in the order they are processed,
        without any regrouping or post-processing.
        """
        dex, set_flat_iter = self._create_dex_pipeline(datasets)
        self._execute_pipeline(dex, set_flat_iter, consumer)

    def to_disk(
        self,
        datasets: Sequence[IntervalDataset],
        output_paths: Sequence[str],
    ) -> None:
        """Process datasets and write results to zarr files with direct writing.

        Results are written directly to final zarr files using lock-based initialization
        and rank-specific output mapping for efficient parallel processing.
        """
        if not datasets:
            raise ValueError("At least one dataset is required")
        if not output_paths:
            raise ValueError("At least one output path is required")
        if len(datasets) != len(output_paths):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of output paths ({len(output_paths)})"
            )

        # Create pipeline and get streaming indices iterator
        dex, set_flat_iter = self._create_dex_pipeline(datasets)
        rank_indices_iterator = dex.simulate(set_flat_iter.indices(), self.batch_size)

        # Create direct zarr consumer with streaming indices
        zarr_consumer = self._create_direct_zarr_consumer(
            output_paths, rank_indices_iterator
        )

        # Execute pipeline with direct writing
        self._execute_pipeline(dex, set_flat_iter, zarr_consumer)

    def _create_dex_pipeline(
        self, datasets: Sequence[IntervalDataset]
    ) -> tuple[Dex, SetFlatIter]:
        """Create and configure the Dex pipeline for processing datasets."""
        fasta_path = self._validate_fasta_paths(datasets)
        dex = GenomicDex.create_pipeline(self.model, fasta_path)
        set_flat_iter = SetFlatIter(datasets)
        return dex, set_flat_iter

    def _execute_pipeline(
        self, dex: Dex, set_flat_iter: SetFlatIter, consumer: ConsumerFn[Tensor]
    ) -> None:
        """Execute the Dex pipeline with the given consumer."""
        dex.execute(
            data=set_flat_iter,
            consumer_fn=consumer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _validate_fasta_paths(self, datasets: Sequence[IntervalDataset]) -> Path | None:
        """Validate that all datasets have the same FASTA path."""
        fasta_path: Path | None = None
        for dataset in datasets:
            dataset_fasta = dataset.associated_fasta_path
            if fasta_path is None:
                fasta_path = dataset_fasta
            elif fasta_path != dataset_fasta:
                raise ValueError("All datasets must have the same FASTA path")
        return fasta_path

    def _create_direct_zarr_consumer(
        self,
        output_paths: Sequence[str],
        rank_indices_iterator: Iterator[Iterable[tuple[int, int]]],
    ) -> ConsumerFn[Tensor]:
        """Create a consumer that writes directly to final zarr files with streaming indices."""

        # Convert torch dtype to zarr-compatible string
        model_dtype = self.model.embed_dtype
        zarr_dtype = str(model_dtype)[6:]  # "torch.float32" -> "float32"

        zarr_writer = MultiZarrWriter(
            output_paths=output_paths,
            shape=(0, self.embed_dim),
            chunks=(self.batch_size, self.embed_dim),
            dtype=zarr_dtype,
        )

        def zarr_consumer(embeddings: Iterable[Tensor]) -> None:
            # Get the indices iterator for this batch
            try:
                batch_indices_iterator = next(rank_indices_iterator)
            except StopIteration:
                raise RuntimeError("No more indices available from rank simulation")

            # Materialize only the indices for this batch
            batch_indices = list(batch_indices_iterator)

            # Convert tensors to numpy arrays and write using the dedicated zarr writer
            numpy_embeddings = [tensor.cpu().numpy() for tensor in embeddings]
            zarr_writer.write_batch(numpy_embeddings, batch_indices)

        return zarr_consumer
