import os
from pathlib import Path


def fix_bed_ext[T: str | Path](path: T) -> T:
    """
    Fixes the extension to either be .bed or .bed.gz
    depending on what exists. Tries adding/removing .gz
    or adding .bed to make it exist.

    @returns same type in

    Throws if no permutation exists.
    """

    wants_str = isinstance(path, str)

    if not wants_str:
        path = str(path)

    if not path.endswith(".bed.gz") and not path.endswith(".bed"):
        path += ".bed.gz"

    if not os.path.isfile(path):
        if path.endswith(".bed.gz"):
            path = path[:-3]
        elif path.endswith(".bed"):
            path += ".gz"

    if os.path.isfile(path):
        if wants_str:
            return path
        return Path(path)

    raise FileNotFoundError(path)
