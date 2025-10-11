import contextlib
import os
import shutil
from dataclasses import dataclass
from typing import final

import numpy as np
import zarr
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
    embeddings. Can be zarr-backed or in-memory (numpy array).
    """

    def __init__(
        self,
        meta: EmbedMeta,
        *,
        data: np.ndarray | None = None,
        data_path: str | None = None,
    ):
        """
        Should not be used directly: create with write_meta for zarr-backed
        Embeds, or by passing an in-memory numpy array.

        @param data: for an in-memory embedding
        @param data_path: for a zarr-backed embedding directory
        """
        super().__init__(meta.embed_dim, meta.dtype, meta.model_info)

        if data is not None and data_path is not None:
            raise ValueError("Provide either `data` or `data_path`, not both.")
        if data is None and data_path is None:
            raise ValueError("Provide either `data` or `data_path`.")

        self._data: np.ndarray | None = data
        self._zarr_array: zarr.Array | None = None
        self.data_path: str | None = data_path

    @property
    def data(self) -> np.ndarray:
        """
        Returns the embedding data as a numpy array.
        If zarr-backed, this will load the data from zarr on first access.
        """
        if self._data is None:
            if self.data_path is None:
                # This state should be unreachable due to __init__ checks
                raise RuntimeError("Embed is not initialized correctly.")

            # Load from zarr directory
            if self._zarr_array is None:
                zarr_store = zarr.open(self.data_path, mode="r")
                if isinstance(zarr_store, zarr.Array):
                    self._zarr_array = zarr_store
                else:
                    raise ValueError(f"Expected zarr.Array, got {type(zarr_store)}")
            self._data = np.asarray(self._zarr_array)
        return self._data

    def unload(self):
        """
        Unloads self.data from memory if it's zarr-backed.
        Accessing the data field after unload() is fine and would trigger reloading.
        For in-memory embeds, this is a no-op.
        """
        if self.data_path is not None:
            # Clear cached data to allow reloading
            self._data = None
            self._zarr_array = None

    def delete(self):
        """
        Deletes the associated zarr directory of this Embed if it is zarr-backed.
        For in-memory embeds, it clears the data from the object.
        This instance should not be used after delete()
        """
        if self.data_path is not None:
            # unload will clear cached data
            self.unload()
            _delete(self.data_path)
        else:
            self._data = None


def write_meta(zarr_path: str, meta: EmbedMeta) -> Embed:
    """
    Should be used often in place of the Embed constructor directly. This
    ensures `Embed`s come with an associated .meta file.
    
    @param zarr_path: Path to the zarr directory
    @param meta: Metadata to associate with the zarr array
    """
    meta_path = zarr_path + ".meta"
    pickle(meta_path, meta)
    return Embed(meta=meta, data_path=zarr_path)


def _check_paths(path: str) -> str:
    if path.endswith(".meta"):
        path = path[:-5]
    
    if not (os.path.isdir(path) and os.path.isfile(f"{path}.meta")):
        raise ValueError(
            "The embed data (zarr directory) cannot be parsed without its metadata (.meta)"
        )

    return path


def parse(path: str) -> Embed:
    """
    @param path: To either the zarr directory (data) or the .meta
    (metadata) file.
    """

    path = _check_paths(path)
    meta: EmbedMeta = unpickle(f"{path}.meta")
    return Embed(meta=meta, data_path=path)


def _delete(path: str):
    """
    @param path: To either the zarr directory (data) or the .meta
    (metadata) file.
    """

    with contextlib.suppress(FileNotFoundError, OSError):
        path = _check_paths(path)
        # Remove zarr directory and metadata file
        if os.path.isdir(path):
            shutil.rmtree(path)
        if os.path.isfile(f"{path}.meta"):
            os.remove(f"{path}.meta")
