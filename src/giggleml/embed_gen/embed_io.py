import contextlib
import os
from dataclasses import dataclass
from typing import cast, final, Optional

import numpy as np
from numpy.typing import DTypeLike

from ..utils.path_pickle import pickle, unpickle


@dataclass
class EmbedMeta:
    embed_dim: int
    dtype: DTypeLike
    model_info: str


@final
class Embed(EmbedMeta):
    """
    Used to store single (vector) embeddings or (tensor collections of
    embeddings. Can be file-backed (memmap) or in-memory (numpy array).
    """

    def __init__(
        self,
        meta: EmbedMeta,
        *,
        data: np.ndarray | None = None,
        data_path: str | None = None,
    ):
        """
        Should not be used directly: create with embedIO.writeMeta for file-backed
        Embeds, or by passing an in-memory numpy array.

        @param data: for an in-memory embedding
        @param dataPath: for a file-backed memmap embedding
        """
        super().__init__(meta.embed_dim, meta.dtype, meta.model_info)

        if data is not None and data_path is not None:
            raise ValueError("Provide either `data` or `dataPath`, not both.")
        if data is None and data_path is None:
            raise ValueError("Provide either `data` or `dataPath`.")

        self._data: np.ndarray | None = data
        self.data_path: str | None = data_path

    @property
    def data(self) -> np.ndarray:
        """
        Returns the embedding data as a numpy array.
        If file-backed, this will load the data as a memmap on first access.
        """
        if self._data is None:
            if self.data_path is None:
                # This state should be unreachable due to __init__ checks
                raise RuntimeError("Embed is not initialized correctly.")

            self._data = cast(
                np.memmap,
                np.memmap(self.data_path, dtype=self.dtype, mode="r").reshape(
                    -1, self.embed_dim
                ),
            )
        return self._data

    def unload(self):
        """
        Unloads self.data from memory if it's a memmap.
        Accessing the data field after unload() is fine and would trigger reloading.
        For in-memory embeds, this is a no-op.
        """
        if self.data_path is not None and self._data is not None:
            if isinstance(self._data, np.memmap):
                self._data.flush()
            # To allow reloading, we must clear the cached data
            self._data = None

    def delete(self):
        """
        Deletes the associated files of this Embed if it is file-backed.
        For in-memory embeds, it clears the data from the object.
        This instance should not be used after delete()
        """
        if self.data_path is not None:
            # unload will set self._data to None
            self.unload()
            _delete(self.data_path)
        else:
            self._data = None


def write_meta(mmap: np.memmap | str, meta: EmbedMeta) -> Embed:
    """
    Should be used often in place of the Embed constructor directly. This
    ensures `Embed`s come with an associated .npy.meta file.
    """

    if not isinstance(mmap, str):
        if mmap.filename is None:
            raise ValueError("mmap is expected to have a non-None filename")
        mmap_path = mmap.filename
    else:
        mmap_path = mmap

    path = mmap_path + ".meta"
    pickle(path, meta)
    return Embed(meta=meta, data_path=mmap_path)


def _check_paths(path: str) -> str:
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

    path = _check_paths(path)
    meta: EmbedMeta = unpickle(f"{path}.npy.meta")
    return Embed(meta=meta, data_path=f"{path}.npy")


def _delete(path: str):
    """
    @param path: To either the .npy (data) or, which must exist, the .npy.meta
    (metadata) file.
    """

    with contextlib.suppress(FileNotFoundError):
        path = _check_paths(path)
        os.remove(f"{path}.npy")
        os.remove(f"{path}.npy.meta")
