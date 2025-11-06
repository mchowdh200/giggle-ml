"""
Takes a set of embeds corresponding to a set of bed files. Embeds per bed file
are mean aggregated the whole of results are represented as a dictionary and
pickled.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import zarr
from numpy.typing import NDArray
from zarr.core.buffer import NDArrayLikeOrScalar

from giggleml.utils.types import PathLike

from ..utils.path_pickle import pickle, unpickle


def build(
    embed_paths: Sequence[PathLike], out_path: PathLike | None = None
) -> dict[str, NDArray]:
    """
    maps each zarr array into an np array by taking the mean along dim zero.
    builds and saves a { zarr -> mean } dict keyed by zarr dir name

    @param embed_paths: paths to zarr arrays
    @param out_path: if None, its inferred as dir/embed_means.pickle where the dir is the first parent dir in the input paths
    """

    if len(embed_paths) == 0:
        raise ValueError("Did you mean to pass zero embed paths?")

    if out_path is None:
        dir = Path(embed_paths[0]).parent
        out_path = dir / "embed_means.pickle"

    mean_embeds: dict[str, NDArray] = dict()

    for path in embed_paths:
        zarr_array = cast(zarr.Array, zarr.open(str(path)))

        if zarr_array.shape[0] == 0:
            continue

        zarr_iter = iter(zarr_array)
        running_sum: NDArrayLikeOrScalar = next(zarr_iter)

        for item in zarr_iter:
            running_sum += item  # pyright: ignore[reportOperatorIssue]

        mean = running_sum / zarr_array.shape[0]  # pyright: ignore[reportOperatorIssue]
        name = Path(path).name
        mean_embeds[name] = mean  # pyright: ignore[reportArgumentType]

    pickle(out_path, mean_embeds)
    return mean_embeds


def parse(path: PathLike) -> dict[str, NDArray]:
    # This is provided for the automatic type cast and semantic usage
    return unpickle(path)
