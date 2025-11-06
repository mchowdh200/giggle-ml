import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen import mean_embed_dict
from giggleml.embed_gen.batch_infer import GenomicEmbedder
from giggleml.models.genomic_model import GenomicModel
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.path_utils import fix_bed_ext
from giggleml.utils.torch_utils import guess_device
from giggleml.utils.types import PathLike


def embed(
    beds_dir: Path,
    out_dir: Path,
    fasta: Path,
    model: GenomicModel,
    batch_size: int,
    num_workers: int,
    limit: int,
) -> dict[str, NDArray]:
    dist.init_process_group()

    try:
        with torch.no_grad():
            rme_paths = [fix_bed_ext(beds_dir / name) for name in rme.bed_names]
            rme_datasets = [
                BedDataset(path, limit=limit, associated_fasta_path=fasta)
                for path in rme_paths
            ]
            engine = GenomicEmbedder(model.to(guess_device()), batch_size, num_workers)
            out_paths = [out_dir / name for name in rme.bed_names]
            engine.to_disk(rme_datasets, out_paths, respect_boundaries=True, log=True)
            return mean_embed_dict.build(out_paths)  # take means
    finally:
        dist.destroy_process_group()


def plot_umap(
    ax: Axes,
    embeddings: np.ndarray,
    labels: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> None:
    """
    Performs UMAP reduction on embeddings and plots the result on a
    given Matplotlib Axes object, colored by labels.

    Args:
        embeddings: The high-dimensional embedding data (n_samples, n_features).
        labels: A list of string labels for each sample (n_samples,).
        ax: The Matplotlib Axes object to plot on.
        title: The title for the subplot.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: The distance metric for UMAP (e.g., 'cosine', 'euclidean').
        random_state: Random state for UMAP reproducibility.
    """

    # 1. Apply UMAP reduction
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embeddings_2d = reducer.fit_transform(embeddings)

    # 2. Create a DataFrame for easy plotting with Seaborn
    df = pd.DataFrame(
        {"UMAP 1": embeddings_2d[:, 0], "UMAP 2": embeddings_2d[:, 1], "label": labels}
    )

    # 3. Plot using Seaborn on the provided Axes
    sns.scatterplot(
        data=df, x="UMAP 1", y="UMAP 2", hue="label", ax=ax, alpha=0.7, s=20
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def umap_combo(embeds_dir: PathLike, out: PathLike, **umap_kwargs):
    dict_path = Path(embeds_dir) / "embed_means.pickle"

    if not os.path.isfile(dict_path):
        print(dict_path)
        means = mean_embed_dict.build(os.listdir(embeds_dir))
    else:
        means = mean_embed_dict.parse(dict_path)

    def simple_cat(cat):
        if any(x in cat.lower() for x in ["brain", "skin", "lung", "digestive"]):
            return cat
        return None

    labels, embeds = zip(*means.items())
    cell_types, chrm_states = zip(*[rme.cell_type_chrm_state_split(x) for x in labels])
    cats = [simple_cat(rme.category_of(x)) for x in cell_types]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    plot_umap(axes[0], embeds, cats, **umap_kwargs)  # pyright: ignore[reportArgumentType]
    plot_umap(axes[1], embeds, chrm_states, **umap_kwargs)  # pyright: ignore[reportArgumentType]
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    rme_dir = Path("data/roadmap_epigenomics")
    beds_dir = rme_dir / "beds"
    out_dir = rme_dir / "embeds"
    fasta = Path("data/hg/hg19.fa")
    model = HyenaDNA("16k")

    embed(beds_dir, out_dir, fasta, model, 32, 8, 50)
    umap_combo(
        out_dir, "experiments/roadmapEpigenomics/hyena_dna_umap.png", n_neighbors=50
    )
