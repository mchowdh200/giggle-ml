from pathlib import Path
from typing import final

import torch

from giggleml.data_wrangling import fasta
from giggleml.utils.types import GenomicInterval

from .dex import Dex
from .embed_model import EmbedModel


class _PicklableCollateWithFASTA:
    """Picklable collate function for FASTA sequence mapping."""

    def __init__(self, fasta_path: Path, model_collate_fn):
        self.fasta_path = fasta_path
        self.model_collate_fn = model_collate_fn

    def __call__(self, batch_items):
        sequences = fasta.map(batch_items, self.fasta_path)
        return self.model_collate_fn(sequences)


class _PicklableCollateNoFASTA:
    """Picklable collate function for direct interval processing."""

    def __init__(self, model_collate_fn):
        self.model_collate_fn = model_collate_fn

    def __call__(self, batch_items):
        return self.model_collate_fn(batch_items)


class _PicklablePreprocessor:
    """Picklable preprocessor for chunking intervals."""

    def __init__(self, max_seq_len: int | None):
        self.max_seq_len = max_seq_len

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

    def __call__(self, interval: GenomicInterval):
        if self.max_seq_len is not None:
            chunks = self._chunk_interval(interval, self.max_seq_len)
            for chunk in chunks:
                yield chunk
        else:
            yield interval


class _PicklablePostprocessor:
    """Picklable postprocessor for averaging chunk embeddings."""

    def __call__(self, embeddings_iter):
        embeddings_list = list(embeddings_iter)
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        else:
            # Average multiple chunk embeddings
            stacked = torch.stack(embeddings_list)
            return torch.mean(stacked, dim=0)


@final
class GenomicDex:
    """Factory for creating genomic processing pipelines using Dex."""

    @staticmethod
    def create_pipeline(model: EmbedModel, fasta_path: Path | None = None) -> Dex:
        """Create a Dex pipeline configured for genomic interval processing."""
        wants_seq = model.wants == "sequences"

        if wants_seq and fasta_path is None:
            raise ValueError("Unable to map to FASTA; missing associated FASTA path")

        # Create pipeline components
        preprocessor = _PicklablePreprocessor(model.max_seq_len)
        postprocessor = _PicklablePostprocessor()
        collate_fn = (
            _PicklableCollateWithFASTA(fasta_path, model.collate)
            if wants_seq and fasta_path is not None
            else _PicklableCollateNoFASTA(model.collate)
        )

        return Dex(
            model=model,
            preprocessor_fn=preprocessor,
            postprocessor_fn=postprocessor,
            collate_fn=collate_fn,
        )

