import contextlib
import os
from dataclasses import dataclass
from functools import cached_property
from typing import cast, final

import numpy as np
from numpy.typing import DTypeLike

from giggleml.utils.types import lazy

from ..utils.pathPickle import pickle, unpickle


@dataclass
class EmbedMeta:
    embedDim: int
    dtype: DTypeLike
    modelInfo: str


@final
@lazy
class Embed(EmbedMeta):
    """
    Used to store single (vector) embeddings or (tensor collections of
    embeddings.
    """

    def __init__(self, dataPath: str, meta: EmbedMeta):
        """
        Should not be used directly: create with embedIO.writeMeta
        """
        super().__init__(meta.embedDim, meta.dtype, meta.modelInfo)
        self.dataPath: str = dataPath

    @cached_property
    def data(self) -> np.memmap:
        return cast(
            np.memmap,
            np.memmap(self.dataPath, dtype=self.dtype, mode="r").reshape(
                -1, self.embedDim
            ),
        )

    def unload(self):
        """
        Unloads self.data (memmap) from memory.
        Accessing the data field after unload() is fine and would trigger reparsing.
        """
        self.data.flush()  # probably unnecessary
        del self.data

    def delete(self):
        """
        Deletes the associated files of this Embed.
        This instance should not be used after delete()
        """
        self.unload()
        _delete(self.dataPath)


def writeMeta(mmap: np.memmap | str, meta: EmbedMeta) -> Embed:
    """
    Should be used often in place of the Embed constructor directly. This
    ensures `Embed`s come with an associated .npy.meta file.
    """

    if not isinstance(mmap, str):
        if mmap.filename is None:
            raise ValueError("mmap is expected to have a non-None filename")
        mmap = mmap.filename

    path = mmap + ".meta"
    pickle(path, meta)
    return Embed(mmap, meta)


def _checkPaths(path: str) -> str:
    if path.endswith(".npy"):
        path = path[:-4]
    elif path.endswith(".npy.meta"):
        path = path[:-9]
    else:
        raise ValueError("Expected path to be either .npy or .npy.meta")

    if not (os.path.isfile(f"{path}.npy") and os.path.isfile(f"{path}.npy.meta")):
        raise ValueError(
            "The embed data (.npy) cannot be parsed without its metadata (.npy.meta)"
        )

    return path


def parse(path: str) -> Embed:
    """
    @param path: To either the .npy (data) or, which must exist, the .npy.meta
    (metadata) file.
    """

    path = _checkPaths(path)
    meta: EmbedMeta = unpickle(f"{path}.npy.meta")
    return Embed(f"{path}.npy", meta)


def _delete(path: str):
    """
    @param path: To either the .npy (data) or, which must exist, the .npy.meta
    (metadata) file.
    """

    with contextlib.suppress(FileNotFoundError):
        path = _checkPaths(path)
        os.remove(f"{path}.npy")
        os.remove(f"{path}.npy.meta")
