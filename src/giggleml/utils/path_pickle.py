import pickle as _pickle
from typing import Any


def pickle(path: str, data: Any):
    with open(path, "wb") as file:
        _pickle.dump(data, file)


def unpickle(path: str):
    with open(path, "rb") as file:
        return _pickle.load(file)
