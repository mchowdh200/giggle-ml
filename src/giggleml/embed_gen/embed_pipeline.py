import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from os.path import exists, isdir
from typing import overload, override

import numpy as np
import torch
from torch import Tensor

from giggleml.embed_gen.batch_infer import BatchInfer
from giggleml.utils.parallel import Parallel
from giggleml.utils.types import lazy

from ..data_wrangling import fasta
from ..data_wrangling.interval_dataset import IntervalDataset
from . import embed_io
from .embed_io import Embed, EmbedMeta
from .embed_model import EmbedModel


@lazy
class EmbedPipeline(ABC):
    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
    ) -> Sequence[Embed]: ...

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: None = None,
    ) -> Tensor: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
    ) -> Tensor: ...

    @abstractmethod
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
    ) -> Tensor | Sequence[Embed] | Embed: ...


class DirectPipeline(EmbedPipeline):
    def __init__(
        self,
        embed_model: EmbedModel,
        batch_size: int,
        worker_count: int | None = None,
        sub_workers: int = 0,
    ):
        """
        @param sub_workers: Number of DataLoader workers for batch construction
        """
        self.model: EmbedModel = embed_model
        self.batch_size: int = batch_size
        self.worker_count: int | None = worker_count
        self.sub_workers: int = sub_workers
        self.infer: BatchInfer = BatchInfer(embed_model, batch_size, sub_workers)

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
    ) -> Sequence[Embed]: ...

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: None = None,
    ) -> Tensor: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: None = None,
    ) -> Tensor: ...

    @override
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str | None = None,
    ) -> Tensor | Sequence[Embed] | Embed:
        # Handle input validation for file-based mode
        if out is not None:
            if isinstance(intervals, Sequence) == isinstance(out, str):
                raise ValueError(
                    "Expecting either both or neither of data & out to be sequences"
                )

        single_input = not isinstance(intervals, Sequence)

        if not isinstance(intervals, Sequence):
            intervals = [intervals]
        if isinstance(out, str):
            out = [out]

        # File existence check only for file-based mode (zarr creates directories)
        if out is not None:
            for out_path in out:
                if exists(out_path):
                    if isdir(out_path):
                        print(
                            f"Output already exists ({out_path}). Continuing interrupted jobs not yet supported.",
                            file=sys.stderr,
                        )

        # Ensure FASTA files are available if needed
        if self.model.wants == "sequences":
            for datum in intervals:
                fa_path = datum.associated_fasta_path
                if fa_path is not None:
                    fasta.ensure_fa(fa_path)

        if out is None:
            # In-memory mode: collect embeddings and return as tensors
            collected_embeddings = []

            def embedding_collector(embeddings: Iterable[Tensor]) -> None:
                for embedding in embeddings:
                    collected_embeddings.append(embedding)

            self.infer.raw(intervals, embedding_collector)

            if single_input:
                return torch.stack(collected_embeddings)
            else:
                # Group embeddings by dataset based on the order they were processed
                # For now, return all as a single tensor - this may need refinement
                return torch.stack(collected_embeddings)

        # File-based mode: use distributed zarr writing
        # Run in distributed environment with picklable worker function
        Parallel(self.worker_count)(
            BatchInfer(self.model, self.batch_size, self.sub_workers).to_disk,
            intervals,
            out,
        )

        # Create Embed references with metadata for zarr directories
        meta = EmbedMeta(self.model.embed_dim, np.float32, str(self.model))
        out_embeds = [embed_io.write_meta(path, meta) for path in out]

        if single_input:
            return out_embeds[0]
        return out_embeds
