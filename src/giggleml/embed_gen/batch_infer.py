import contextlib
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, final, override

import torch
import torch.distributed as dist
from torch import Tensor, nn

from giggleml.data_wrangling import fasta
from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.embed_gen.in_dex import InDex
from giggleml.embed_gen.multi_zarr_writer import MultiZarrWriter
from giggleml.iter_utils.set_flat_iter import SetFlatIter
from giggleml.models.genomic_model import GenomicModel
from giggleml.utils.print_utils import progress_logger
from giggleml.utils.torch_utils import get_world_size
from giggleml.utils.types import GenomicInterval, IntInt, PathLike, lazy
from giggleml.utils.utils.collection_utils import as_list

from .dex import (
    ConsumerFn,
)

type EmbedConsumer = Callable[[Sequence[tuple[IntInt, Tensor]]], None]


type _InDex = InDex[
    GenomicInterval | None,
    GenomicInterval | None,
    GenomicInterval | None,
    Tensor | None,
    Any,
    Any,
]


@lazy
@final
class GenomicEmbedder:
    def __init__(
        self,
        model: GenomicModel,
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
        consumer: EmbedConsumer,
        respect_boundaries: bool = False,
        auto_reclaim: bool = False,
    ) -> None:
        """Process datasets with a generic consumer function.

        The consumer receives individual tensors in the order they are processed,
        without any regrouping or post-processing.

        @arg respect_boundaries: If True, returned blocks will not span multiple chunks
        @arg auto_reclaim: If True, auto move outputs to cpu
        """

        dex, set_flat_iter = self._create_dex_pipeline(datasets, respect_boundaries)
        indices = list(set_flat_iter.indices())

        def raw_consumer_wrapper(output: list[tuple[int, Tensor | None]]) -> None:
            nonlocal indices
            clean = self._clean_batch(indices, output)
            if clean:
                return consumer(clean)

        self._execute_pipeline(dex, set_flat_iter, raw_consumer_wrapper, auto_reclaim)

    def to_disk(
        self,
        datasets: Sequence[IntervalDataset],
        output_paths: Sequence[PathLike],
        respect_boundaries: bool = True,
        log: bool = True,
    ) -> None:
        """Process datasets and write results to zarr files with direct writing.

        Results are written directly to final zarr files. The new batching
        strategy ensures indices are correctly paired with outputs.

        @arg respect_boundaries: If True, returned blocks will not span multiple chunks
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

        # Create pipeline
        dex, set_flat_iter = self._create_dex_pipeline(datasets, respect_boundaries)
        indices = list(set_flat_iter.indices())

        output_counts = [len(ds) for ds in datasets]
        expected_rank_output_count = round(
            len(set_flat_iter) / self.batch_size / get_world_size()
        )

        try:
            if log:
                with progress_logger(
                    expected_rank_output_count, "embedding", only_on_rank_zero=True
                ) as log_ckpt:
                    zarr_consumer = self._create_direct_zarr_consumer(
                        indices, output_paths, output_counts, log_ckpt
                    )
                    self._execute_pipeline(dex, set_flat_iter, zarr_consumer)
                    dist.barrier()
            else:
                zarr_consumer = self._create_direct_zarr_consumer(
                    indices, output_paths, output_counts, lambda: None
                )
                self._execute_pipeline(dex, set_flat_iter, zarr_consumer)
        finally:
            # clean up residual locks. supposed to happen naturally, but not in
            # case of interrupted job
            for out in output_paths:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(f"{str(out)}.init.lock")
                with contextlib.suppress(FileNotFoundError):
                    os.remove(f"{str(out)}.resize.lock")

    def _create_dex_pipeline(
        self,
        datasets: Sequence[IntervalDataset],
        respect_boundaries: bool = True,
    ) -> tuple[_InDex, SetFlatIter[GenomicInterval]]:
        """Create and configure the Dex pipeline for processing datasets."""
        fasta_path = self._validate_fasta_paths(datasets)

        wants_seq = self.model.wants == "sequences"

        if wants_seq and fasta_path is None:
            raise ValueError("Unable to map to FASTA; missing associated FASTA path")

        # Create pipeline components
        preprocessor = _Preprocessor(self.model.max_seq_len)
        collate_fn = _Collate(fasta_path if wants_seq else None, self.model.collate)
        wrapped_model = _ModelWrap(self.model)
        postprocessor = _Postprocessor()
        dex = InDex(
            module=wrapped_model,
            preprocessor_fn=preprocessor,
            postprocessor_fn=postprocessor,  # pyright: ignore[reportArgumentType]
            collate_fn=collate_fn,
        )
        set_flat_iter = SetFlatIter(
            datasets, round_to_multiple=self.batch_size if respect_boundaries else 1
        )
        return dex, set_flat_iter

    def _execute_pipeline(
        self,
        dex: _InDex,
        set_flat_iter: SetFlatIter[GenomicInterval],
        consumer: Callable[[list[tuple[int, Tensor | None]]], None],
        auto_reclaim: bool = True,
    ) -> None:
        """Execute the Dex pipeline with the given consumer."""

        dex.execute(
            data=set_flat_iter,
            consumer_fn=consumer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            auto_reclaim=auto_reclaim,
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
        true_indices: Sequence[IntInt | None],
        output_paths: Sequence[PathLike],
        output_counts: Sequence[int],
        log_ckpt: Callable[[], None],
    ) -> ConsumerFn[tuple[int, Tensor | None]]:
        """
        Create a consumer that writes directly to final zarr files.
        It now receives (index_batch, output_batch) tuples directly.
        """

        # Convert torch dtype to zarr-compatible string
        zarr_dtype = str(self.model.embed_dtype)[6:]  # "torch.float32" -> "float32"
        zarr_writer = MultiZarrWriter(
            output_paths=output_paths,
            shape=(0, self.embed_dim),
            initial_lengths=output_counts,
            chunks=(self.batch_size, self.embed_dim),
            dtype=zarr_dtype,
            # grow_size=self.batch_size * get_world_size() * 2,
        )

        def zarr_consumer(batch: list[tuple[int, Tensor | None]]) -> None:
            nonlocal true_indices
            indices, embeddings = zip(*self._clean_batch(true_indices, batch))

            if len(indices) != 0:
                numpy_embeddings = [t.cpu().numpy() for t in embeddings]
                zarr_writer.write_batch(numpy_embeddings, indices)
                log_ckpt()

        return zarr_consumer

    @as_list
    def _clean_batch(
        self,
        true_indices: Sequence[IntInt | None],
        batch: Sequence[tuple[int, Tensor | None]],
    ) -> Iterator[tuple[IntInt, Tensor]]:
        for idx, tensor in batch:
            true_idx = true_indices[idx]

            if tensor is not None and true_idx is not None:
                yield true_idx, tensor


# INFO --------------------------
#        Generic Pipeline
# -------------------------------


class _Preprocessor:
    """chunking intervals."""

    def __init__(self, max_seq_len: int | None):
        self.max_seq_len: int | None = max_seq_len

    def _chunk_interval(
        self, interval: GenomicInterval, max_size: int
    ) -> list[GenomicInterval]:
        """Split an interval into chunks of max_size."""
        chrom, start, end = interval
        length = end - start

        if length <= max_size:
            return [interval]

        chunks = []
        current_start = start
        while current_start < end:
            current_end = min(current_start + max_size, end)
            chunks.append((chrom, current_start, current_end))
            current_start = current_end

        return chunks

    def __call__(self, interval: GenomicInterval | None):
        if interval is not None and self.max_seq_len is not None:
            chunks = self._chunk_interval(interval, self.max_seq_len)
            for chunk in chunks:
                yield chunk
        else:
            yield interval


class _Collate:
    """FASTA sequence mapping."""

    def __init__(self, fasta_path: Path | None, model_collate_fn: Callable):
        self.fasta_path: Path | None = fasta_path
        self.model_collate_fn = model_collate_fn

    def __call__(self, batch: Iterable[GenomicInterval | None]):
        total_intervals = batch
        clean_intervals = [x for x in total_intervals if x is not None]

        if len(clean_intervals) == 0:
            return None

        if self.fasta_path:
            processed_batch = fasta.map(clean_intervals, self.fasta_path)
        else:
            processed_batch = clean_intervals

        return self.model_collate_fn(processed_batch)


class _ModelWrap(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = model

    @override
    def forward(self, batch):
        if batch is not None:
            return self.model(batch)
        return [None]


class _Postprocessor:
    """averaging chunk embeddings."""

    def __call__(self, embeddings: list[Tensor | None]):
        # Average multiple chunk embeddings
        clean_embeds = [x for x in embeddings if x is not None]

        if len(clean_embeds) == 0:
            return None

        stacked = torch.stack(clean_embeds)
        return torch.mean(stacked, dim=0)
