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

from ..embedGen import embedIO
from ..utils.pathPickle import pickle, unpickle

EmbedDict = dict[str, np.ndarray]


def build(embedPaths: Sequence[str], outPath: str | None = None) -> EmbedDict:
    """
    @param embedPaths: a collection of .npy files holding
    @param outPath: if None, its inferred as dirname(embedPaths)/master.pickle

    Keys in the result dict do not have the .npy extension.
        Eg, "x/y/z/item.npy" -> "item"
    """

    if len(embedPaths) == 0:
        raise ValueError("Did you mean to pass zero embed paths?")

    for path in embedPaths:
        if not path.endswith(".npy"):
            raise ValueError("Expecting embedPaths to be a collection .npy files")

    if outPath is None:
        for i in range(len(embedPaths)):
            if dirname(embedPaths[i]) != dirname(embedPaths[i - 1]):
                raise ValueError(
                    "Unable to infer outPath if embedPaths have different parent directories"
                )
        outPath = dirname(embedPaths[0]) + "/master.pickle"

    meanEmbeds: list[np.ndarray] = list()

    for path in embedPaths:
        embed = embedIO.parse(path).data

        if len(embed.shape) == 1:
            # Case A:
            # we're dealing with a single embed, from a lone interval, which is
            # unexpected because we were expecting to deal with embed sets
            print(
                f"An embed path corresponds to an embed produced by a single interval as "
                "opposed to an embed set. ({path})",
                file=sys.stderr,
            )
            meanEmbeds.append(embed)
        else:
            # Case B: usual case
            # embed should be shape (N, embedDim)
            meanEmbed = embed.mean(axis=0)
            meanEmbeds.append(meanEmbed)

    # x/y/z/basename.npy -> basename
    names = [basename(path)[:-4] for path in embedPaths]

    final = {name: embed for name, embed in zip(names, meanEmbeds)}
    pickle(outPath, final)
    return final


def parse(path: str) -> EmbedDict:
    # This is provided for the automatic type cast and semantic usage
    return unpickle(path)
