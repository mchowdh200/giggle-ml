import os
from dataclasses import dataclass
from typing import cast, final

import numpy as np
from numpy.typing import DTypeLike

from ..utils.pathPickle import pickle, unpickle


@dataclass
class EmbedMeta:
    embedDim: int
    dtype: DTypeLike
    modelInfo: str


@final
class Embed(EmbedMeta):
    """
    Used to store single (vector) embeddings or (tensor collections of
    embeddings.
    """

    def __init__(self, data: np.memmap, meta: EmbedMeta):
        """
        Should not be used directly: create with embedIO.writeMeta
        """
        super().__init__(meta.embedDim, meta.dtype, meta.modelInfo)
        self.data = data


def writeMeta(mmap: np.memmap, meta: EmbedMeta) -> Embed:
    """
    Should be used often in place of the Embed constructor directly. This
    ensures `Embed`s come with an associated .npy.meta file.
    """

    if mmap.filename is None:
        raise ValueError("mmap is expected to have a non-None filename")

    path = mmap.filename + ".meta"
    pickle(path, meta)
    return Embed(mmap, meta)


def parse(path: str) -> Embed:
    """
    @param path: To either the .npy (data) or, which must exist, the .npy.meta
    (metadata) file.
    """

    if path.endswith(".npy"):
        path = path[:-4]
    elif path.endswith(".npy.meta"):
        path = path[:-9]
    else:
        raise ValueError("Expected path to be either .npy or .npy.meta")

    if not (os.path.isfile(f"{path}.npy") and os.path.isfile(f"{path}.npy.meta")):
        raise ValueError("The embed data (.npy) cannot be parsed without its metadata (.npy.meta)")

    meta: EmbedMeta = unpickle(f"{path}.npy.meta")
    data = np.memmap(f"{path}.npy", dtype=meta.dtype, mode="r")
    data: np.memmap = cast(np.memmap, data.reshape(-1, meta.embedDim))
    return Embed(data, meta)
