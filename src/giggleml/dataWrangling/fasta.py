from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

import pyfastx

from ..utils.types import GenomicInterval
from .intervalDataset import IntervalDataset
from .listDataset import ListDataset

Fasta = dict[str, str]  # chromosome -> sequence map
knownFa: dict[str, Fasta] = dict()
cache: tuple[str, Fasta] = ("", dict())


def ensureFa(fastaPath: str) -> Fasta:
    """
    Returns a chromosome -> sequence map based on
    1) memory cache
    2) fastx idx
    3) parses the fasta file (with fastx & caches)
    """
    global cache

    # attempt to skip the str(Path(.).resolve()) & dict call.
    # makes duplicate calls free
    if fastaPath == cache[0]:
        return cache[1]

    fastaPathNormalized = str(Path(fastaPath).resolve())  # normalization

    if fastaPathNormalized not in knownFa:
        idx = pyfastx.Fasta(fastaPathNormalized)
        seqs: Fasta = {seq.name: seq.seq for seq in idx}
        knownFa[fastaPathNormalized] = seqs

    result = knownFa[fastaPathNormalized]
    cache = (fastaPath, result)
    return result


@overload
def map(data: IntervalDataset) -> ListDataset[str]: ...


@overload
def map(data: Sequence[GenomicInterval], fastaPath: str) -> list[str]: ...


# TODO: inference pipeline should expose a parameter to avoid using fasta cache
def map(
    data: IntervalDataset | Sequence[GenomicInterval], fastaPath: str | None = None
) -> list[str] | ListDataset[str]:
    """
    @param fastaPath: can be a .fa or .gz file
    """

    if not isinstance(data, Sequence):
        fastaPath = data.associatedFastaPath
    if fastaPath is None:
        raise ValueError("Fasta path not provided")

    seqs = ensureFa(fastaPath)
    results: list[Any] = [None] * len(data)

    # can't use enumerate here because Datasets aren't always __iter__able
    for i in range(len(data)):
        name, start, end = data[i]

        if name not in seqs:
            raise ValueError(f"Chromosome {name} not found in {fastaPath}")

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
