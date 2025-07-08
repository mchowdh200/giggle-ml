from dataclasses import dataclass, field

import numpy as np
import ruranges
from numpy.typing import NDArray

import giggleml.utils.roadmapEpigenomics as rme
from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.utils.misc import fix_bed_ext
from giggleml.utils.time_this import time_this


@dataclass
class NDSource:
    starts: NDArray[np.int32]
    ends: NDArray[np.int32]
    chrms: NDArray[np.uint32]
    file_idx: NDArray[np.int32]


@dataclass
class DataSource:
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    chrms: list[int] = field(default_factory=list)
    file_idx: list[int] = field(default_factory=list)

    def to_numpy(self):
        dtype = np.int32
        return NDSource(
            np.array(self.starts, dtype=dtype),
            np.array(self.ends, dtype=dtype),
            np.array(self.chrms, dtype=np.uint32),
            np.array(self.file_idx, dtype=dtype),
        )


def main():
    cache = dict()

    def fetch(i):
        if i not in cache:
            path = fix_bed_ext(f"data/roadmap_epigenomics/beds/{bed}.bed.gz")
            cache[i] = list(iter(BedDataset(path)))
        return cache[i]

    chromosomes = [*[f"chr{x}" for x in range(1, 24)], "chrM", "chrX", "chrY"]

    def stack(source, i):
        data = DataSource()

        for chrm, start, end in source:
            data.starts.append(start)
            data.ends.append(end)
            data.chrms.append(chromosomes.index(chrm))
            data.file_idx.append(i)

        return data

    wet_database = DataSource()

    with time_this("load source"):
        for i, bed in enumerate(rme.bedNames):
            data = stack(fetch(i), i)
            wet_database.starts.extend(data.starts)
            wet_database.ends.extend(data.ends)
            wet_database.chrms.extend(data.chrms)
            wet_database.file_idx.extend(data.file_idx)

        database = wet_database.to_numpy()

    for i in range(10):
        query = stack(fetch(i), i).to_numpy()

        with time_this("ruranges"):
            print(" ..", i)
            results = ruranges.overlaps(
                starts=database.starts,
                ends=database.ends,
                groups=database.chrms,
                starts2=query.starts,
                ends2=query.ends,
                groups2=query.chrms,
            )

    # ft.main()


if __name__ == "__main__":
    main()
