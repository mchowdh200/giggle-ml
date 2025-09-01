import math
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypedDict, final, override

from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.utils.types import GenomicInterval


def chunk_and_split[T](
    items: Sequence[T], chunk_size: int, splits: Iterable[float]
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

    for i in range(0, len(items), chunk_size):
        j = min(i + chunk_size, len(items))
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
        self, data: dict[tuple[str, str], Sequence[Group]], groups_per_item: int
    ):
        self.groups = data
        self.groups_per_item = groups_per_item

    @override
    def __iter__(self) -> Iterator[list[Group]]:
        while True:
            random_key = random.choice(list(self.groups.keys()))
            random_category = self.groups[random_key]
            item = random.sample(random_category, k=self.groups_per_item)
            yield item


class TTV[T](TypedDict):
    train: T
    test: T
    validation: T


def make_datasets(
    rme_beds: str,
    hg: str,
    group_size: int,
    groups_per_item: int,
    train_split: float,
    test_split: float,
) -> TTV[RoadEpGroupsDataset]:
    """
    @param rmeBedsPath: the path to the beds dir, eg: data/roadmap_epigenomics/beds
    @param hg: the path to the hg file, eg: hg/hg19.fa
    @param groupsPerItem: the amount of groups per Dataset#__iter__#next item
    @param trainSplit: .8 indicates 80% is for training
    @param testSplit: .5 indicates that the test set receives 50% of the (test + validation) set
    """

    rme_beds_path = Path(rme_beds)
    hg_path = Path(hg)

    if not rme_beds_path.exists():
        raise ValueError(f"{rme_beds} must exist.")
    if not hg_path.exists():
        raise ValueError(f"{hg} must exist.")

    # (train/test/validation) -> label (x,y) -> list[Group]
    parts: TTV[dict[tuple[str, str], Sequence[Group]]] = {
        "train": dict(),
        "test": dict(),
        "validation": dict(),
    }

    threshold = math.ceil(
        group_size * groups_per_item / (1 - train_split) / (1 - test_split)
    )
    print(f"Using a minimum size of {threshold} per bed file")

    for cell_type in rme.cell_types:
        for chrm_state in rme.chromatin_states:
            label = f"{cell_type}_{chrm_state}"
            path = rme_beds_path / f"{label}.bed"

            if not path.exists():
                path = rme_beds_path / f"{label}.bed.gz"
            if not path.exists():
                raise ValueError(f"Missing expected file {label}.bed[.gz]")

            dataset = BedDataset(str(path), str(hg_path))

            if len(dataset) < threshold:
                print(f"Skipping {label} due to insufficient data.")
                continue

            train, test, validation = chunk_and_split(
                list(iter(dataset)), group_size, [train_split, test_split]
            )

            parts["train"][cell_type, chrm_state] = train
            parts["test"][cell_type, chrm_state] = test
            parts["validation"][cell_type, chrm_state] = validation

    return {
        "train": RoadEpGroupsDataset(parts["train"], groups_per_item),
        "test": RoadEpGroupsDataset(parts["test"], groups_per_item),
        "validation": RoadEpGroupsDataset(parts["validation"], groups_per_item),
    }
