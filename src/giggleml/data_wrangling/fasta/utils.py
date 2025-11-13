from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

import pyfastx

from giggleml.data_wrangling.interval_dataset import IntervalDataset
from giggleml.data_wrangling.list_dataset import ListDataset
from giggleml.utils.types import GenomicInterval, PathLike

Fasta = dict[str, str]  # chromosome -> sequence map
known_fa: dict[str, Fasta] = dict()
cache: tuple[str, Fasta] = ("", dict())


def ensure_fa(path: PathLike) -> Fasta:
    """
    Returns a chromosome -> sequence map based on
    1) memory cache
    2) fastx idx
    3) parses the fasta file (with fastx & caches)
    """
    global cache
    normal_path = str(Path(path).resolve())  # normalization

    # attempt to skip the str(Path(.).resolve()) & dict call.
    # makes duplicate calls free
    if normal_path == cache[0]:
        return cache[1]

    if normal_path not in known_fa:
        idx = pyfastx.Fasta(normal_path)
        seqs: Fasta = {seq.name: seq.seq for seq in idx}
        known_fa[normal_path] = seqs

    result = known_fa[normal_path]
    cache = (normal_path, result)
    return result


def shape(path: PathLike) -> dict[str, int]:
    return {k: len(v) for k, v in ensure_fa(path).items()}


@overload
def map(data: IntervalDataset) -> ListDataset[str]: ...


@overload
def map(data: Sequence[GenomicInterval], fasta_path: Path) -> list[str]: ...


# TODO: inference pipeline should expose a parameter to avoid using fasta cache
def map(
    data: IntervalDataset | Sequence[GenomicInterval], fasta_path: Path | None = None
) -> list[str] | ListDataset[str]:
    """
    @param fastaPath: can be a .fa or .gz file
    """

    if not isinstance(data, Sequence):
        fasta_path = data.associated_fasta_path
    if fasta_path is None:
        raise ValueError("Fasta path not provided")

    seqs = ensure_fa(fasta_path)
    results: list[Any] = [None] * len(data)

    # can't use enumerate here because Datasets aren't always __iter__able
    for i in range(len(data)):
        name, start, end = data[i]

        if name not in seqs:
            raise ValueError(f"Chromosome {name} not found in {fasta_path}")

        chrm = seqs[name]

        if end - start > len(chrm):
            raise ValueError(
                f"Interval ({start}, {end}) exceeds length {len(chrm)} chromosome {name}"
            )

        item = chrm[start:end]
        results[i] = item

    if isinstance(data, Sequence):
        return results

    return ListDataset[str](results)
