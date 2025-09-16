import os
from pathlib import Path
from typing import Any, cast, overload


def fix_bed_ext[T: str | Path](path: T) -> T:
    """
    Fixes the extension to either be .bed or .bed.gz
    depending on what exists. Tries adding/removing .gz
    or adding .bed to make it exist.

    @returns same type in

    Throws if no permutation exists.
    """

    out = str(path)

    if not out.endswith(".bed.gz") and not out.endswith(".bed"):
        # no extension? add
        out += ".bed.gz"

    if not os.path.isfile(out):
        # try the other permutation
        if out.endswith(".bed.gz"):
            out = out[:-3]
        elif out.endswith(".bed"):
            out += ".gz"

        # neither is found
        if not os.path.isfile(out):
            raise FileNotFoundError(out)

    # coherent type
    if isinstance(path, str):
        return cast(T, out)
    return cast(T, Path(out))


@overload
def as_path(path: Path | None) -> Path: ...


@overload
def as_path(path: str | None) -> Path: ...


def as_path(path: str | Path | None) -> Path | None:
    if isinstance(path, str):
        return Path(path)
    return path


def is_path_like(path: Any) -> bool:
    """because isinstance(x, PathLike) doesn't work in general case"""
    return isinstance(path, (str, bytes, os.PathLike))
