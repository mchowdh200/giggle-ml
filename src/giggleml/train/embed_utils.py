"""
Utility functions for embedding operations in training.
"""

from pathlib import Path

from giggleml.data_wrangling.interval_dataset import IntervalDataset, MemoryIntervalDataset
from giggleml.embed_gen.embed_pipeline import EmbedPipeline
from giggleml.utils.path_utils import as_path
from giggleml.utils.types import GenomicInterval


# A cluster is a list of bed files
Cluster = list[list[GenomicInterval]]


def embed_batch(
    pipeline: EmbedPipeline, edim: int, batch: list[Cluster], fasta: Path | str
):
    """
    Convert a batch of interval clusters to embeddings.
    
    Args:
        pipeline: The embedding pipeline to use
        edim: Embedding dimension
        batch: List of clusters, each containing lists of genomic intervals
        fasta: Path to FASTA reference file
        
    Returns:
        Tensor of embeddings with shape [batch_size, cluster_size * density, edim]
    """
    fasta = as_path(fasta)
    flat_input: list[IntervalDataset] = list()

    for cluster in batch:
        for intervals in cluster:
            dataset = MemoryIntervalDataset(intervals, fasta)
            flat_input.append(dataset)

    embeds = pipeline.embed(flat_input)
    return embeds.reshape(len(batch), -1, edim)