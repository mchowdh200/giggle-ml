"""
Utility functions for embedding operations in training.
"""

from pathlib import Path

from giggleml.data_wrangling.interval_dataset import (
    IntervalDataset,
    MemoryIntervalDataset,
)
from giggleml.embed_gen.embed_pipeline import EmbedPipeline
from giggleml.train.rme_clusters_dataset import Cluster
from giggleml.utils.path_utils import as_path


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
        for group in cluster:
            dataset = MemoryIntervalDataset(group, fasta)
            flat_input.append(dataset)

    # shape: [ clusters * groups_per_cluster, intervals_per_group, edim ]
    embeds = pipeline.embed(flat_input)
    # shape: [ clusters * groups_per_cluster, edim ]
    embeds = embeds.mean(dim=1)  # centroid embed, per group
    # shape: [ clusters, groups_per_cluster, edim ]
    return embeds.reshape(len(batch), -1, edim)
