"""
Takes a set of embeds corresponding to a set of bed files. Embeds per bed file
are mean aggregated the whole of results are represented as a dictionary and
pickled.
"""

import sys
from collections.abc import Sequence
from os.path import basename, dirname

import numpy as np
import torch

from ..embed_gen import embed_io
from ..utils.path_pickle import pickle, unpickle

EmbedDict = dict[str, np.ndarray]


def build(embed_paths: Sequence[str], out_path: str | None = None) -> EmbedDict:
    """
    @param embedPaths: a collection of .npy files holding
    @param outPath: if None, its inferred as dirname(embedPaths)/master.pickle

    Keys in the result dict do not have the .npy extension.
        Eg, "x/y/z/item.npy" -> "item"
    """

    if len(embed_paths) == 0:
        raise ValueError("Did you mean to pass zero embed paths?")

    for path in embed_paths:
        if not path.endswith(".npy"):
            raise ValueError("Expecting embedPaths to be a collection .npy files")

    if out_path is None:
        for i in range(len(embed_paths)):
            if dirname(embed_paths[i]) != dirname(embed_paths[i - 1]):
                raise ValueError(
                    "Unable to infer outPath if embedPaths have different parent directories"
                )
        out_path = dirname(embed_paths[0]) + "/master.pickle"

    mean_embeds: list[np.ndarray] = list()

    for path in embed_paths:
        embed = embed_io.parse(path).data

        if len(embed.shape) == 1:
            # Case A:
            # we're dealing with a single embed, from a lone interval, which is
            # unexpected because we were expecting to deal with embed sets
            print(
                f"An embed path corresponds to an embed produced by a single interval as "
                "opposed to an embed set. ({path})",
                file=sys.stderr,
            )
            mean_embeds.append(embed)
        else:
            # Case B: usual case
            # embed should be shape (N, embedDim)
            mean_embed = embed.mean(axis=0)
            mean_embeds.append(mean_embed)

    # x/y/z/basename.npy -> basename
    names = [basename(path)[:-4] for path in embed_paths]

    final = {name: embed for name, embed in zip(names, mean_embeds)}
    pickle(out_path, final)
    return final


def parse(path: str) -> EmbedDict:
    # This is provided for the automatic type cast and semantic usage
    return unpickle(path)
