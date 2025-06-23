import math
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypedDict, final, override

from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset

import giggleml.utils.roadmapEpigenomics as rme
from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.utils.types import GenomicInterval


def chunkAndSplit[T](
    items: Sequence[T], chunkSize: int, splits: Iterable[float]
) -> Iterable[list[list[T]]]:
    """
    Chunks items then splits (usually into train/validation/test sets).

    @param splits: a train/validation/test split of 80%, 10%, 10% would be [.8, .5] for two splits,
    three partitions. Taking 80% and 50%, respectively, for the initial partition.

    1. chunks,
    .......... -> [...][...][...][...]

    2. and splits, eg for splits=[.8, .5]:
        * * * * * * * *
        |---------|----|
            80%
                  |-||-|
                 50%   50%
    """

    # TODO: stop upfront chunking

    minis = list[list[T]]()

    for i in range(0, len(items), chunkSize):
        j = min(i + chunkSize, len(items))
        minis.append(list(items[i:j]))

    rest = minis

    for split in splits:
        head, rest = train_test_split(rest, train_size=split, random_state=42)
        yield head

    yield rest


Group = list[GenomicInterval]


@final
class RoadEpGroupsDataset(IterableDataset):
    def __init__(
        self, data: dict[tuple[str, str], Sequence[Group]], groupsPerItem: int
    ):
        self.groups = data
        self.groupsPerItem = groupsPerItem

    @override
    def __iter__(self) -> Iterator[list[Group]]:
        while True:
            randomKey = random.choice(list(self.groups.keys()))
            randomCategory = self.groups[randomKey]
            item = random.sample(randomCategory, k=self.groupsPerItem)
            yield item


class TTV[T](TypedDict):
    train: T
    test: T
    validation: T


def makeDatasets(
    rmeBeds: str,
    hg: str,
    groupSize: int,
    groupsPerItem: int,
    trainSplit: float,
    testSplit: float,
) -> TTV[RoadEpGroupsDataset]:
    """
    @param rmeBedsPath: the path to the beds dir, eg: data/roadmap_epigenomics/beds
    @param hg: the path to the hg file, eg: hg/hg19.fa
    @param groupsPerItem: the amount of groups per Dataset#__iter__#next item
    @param trainSplit: .8 indicates 80% is for training
    @param testSplit: .5 indicates that the test set receives 50% of the (test + validation) set
    """

    rmeBedsPath = Path(rmeBeds)
    hgPath = Path(hg)

    if not rmeBedsPath.exists():
        raise ValueError(f"{rmeBeds} must exist.")
    if not hgPath.exists():
        raise ValueError(f"{hg} must exist.")

    # (train/test/validation) -> label (x,y) -> list[Group]
    parts: TTV[dict[tuple[str, str], Sequence[Group]]] = {
        "train": dict(),
        "test": dict(),
        "validation": dict(),
    }

    threshold = math.ceil(
        groupSize * groupsPerItem / (1 - trainSplit) / (1 - testSplit)
    )
    print(f"Using a minimum size of {threshold} per bed file")

    for cellType in rme.cellTypes:
        for chrmState in rme.chromatinStates:
            label = f"{cellType}_{chrmState}"
            path = rmeBedsPath / f"{label}.bed"

            if not path.exists():
                path = rmeBedsPath / f"{label}.bed.gz"
            if not path.exists():
                raise ValueError(f"Missing expected file {label}.bed[.gz]")

            dataset = BedDataset(str(path), str(hgPath))

            if len(dataset) < threshold:
                print(f"Skipping {label} due to insufficient data.")
                continue

            train, test, validation = chunkAndSplit(
                list(iter(dataset)), groupSize, [trainSplit, testSplit]
            )

            parts["train"][cellType, chrmState] = train
            parts["test"][cellType, chrmState] = test
            parts["validation"][cellType, chrmState] = validation

    return {
        "train": RoadEpGroupsDataset(parts["train"], groupsPerItem),
        "test": RoadEpGroupsDataset(parts["test"], groupsPerItem),
        "validation": RoadEpGroupsDataset(parts["validation"], groupsPerItem),
    }
