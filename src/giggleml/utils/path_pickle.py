import pickle as _pickle
from typing import Any

from giggleml.utils.types import PathLike


def pickle(path: PathLike, data: Any):
    with open(path, "wb") as file:
        _pickle.dump(data, file)


def unpickle(path: PathLike):
    with open(path, "rb") as file:
        return _pickle.load(file)
