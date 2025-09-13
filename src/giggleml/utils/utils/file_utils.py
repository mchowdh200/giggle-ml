from os import PathLike
from pathlib import Path


def file_ext(file: PathLike):
    """the final file suffix, excluding the period"""
    return Path(file).suffix[1:]
