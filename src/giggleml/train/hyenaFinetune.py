import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from random import randint
from typing import final

import pytorch_lightning as pl
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import IterableDataset

import giggleml.utils.roadmapEpigenomics as rme
from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.embedGen.embedModel import EmbedModel
from giggleml.utils.types import GenomicInterval


@final
class EmbeddingFineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module to fine-tune the HyenaDNA embedding model
    using triplet loss.
    """

    def __init__(
        self, model: EmbedModel, learningRate: float = 2e-5, margin: float = 1.0
    ):
        super().__init__()
        self.model = model
        self.learningRate = learningRate
        # Triplet loss requires a margin
        self.loss = nn.TripletMarginLoss(margin=margin)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model.embed(batch)

    def training_step(self, batch, batch_idx):
        # The batch is expected to be a dictionary containing
        # anchor, positive, and negative input_ids.
        anchor_ids = batch["anchor_input_ids"]
        positive_ids = batch["positive_input_ids"]
        negative_ids = batch["negative_input_ids"]

        # Generate embeddings for the triplet
        anchor_embedding = self(anchor_ids)
        positive_embedding = self(positive_ids)
        negative_embedding = self(negative_ids)

        # Calculate the triplet loss
        loss = self.loss(anchor_embedding, positive_embedding, negative_embedding)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        # Use the AdamW optimizer, which is common for Transformer-based models
        optimizer = AdamW(self.parameters(), lr=self.learningRate)
        return optimizer

    # You can optionally add validation_step and test_step methods
    # if you have validation and test datasets.


# INFO: dataset prep


def makeClusters[T](
    items: Sequence[T], clusterSize: int, splits: Iterable[float]
) -> Iterable[Sequence[T]]:
    """
    Chunks items into mini batches and then splits (usually into train/validation/test sets).

    @param splits: a train/validation/test split of 80%, 10%, 10% would be [.8, .5] for two splits,
    three partitions. Taking 80% and 50%, respectively, for the initial partition.

        * * * * * * * *
        |---------|----|
            80%
                  |-||-|
                 50%   50%
    """

    minis = list[Sequence[T]]()

    for i in range(0, len(items), clusterSize):
        j = min(i, len(items))
        minis.append(items[i:j])

    rest = minis

    for split in splits:
        head, rest = train_test_split(rest, train_size=split, random_state=42)
        yield head


type ClusterList = list[list[GenomicInterval]]


@final
class RoadEpTriplets(IterableDataset):
    def __init__(self, rmeBeds: str, hg: str, clusterSize: int):
        """
        @param rmeBeds: the path to the beds dir, eg: data/roadmap_epigenomics/beds
        @param hg: the path to the hg file, eg: hg/hg19.fa
        """

        self.rmeBeds = Path(rmeBeds)
        self.hg = Path(hg)

        if not self.rmeBeds.exists():
            raise ValueError(f"{rmeBeds} must exist.")
        if not self.hg.exists():
            raise ValueError(f"{hg} must exist.")

    def _makeClusters(self, clusterSize) -> dict[str, dict[str, ClusterList]]:
        splits = [0.8, 0.5]
        # map:  (train/validation/test) -> label -> list[cluster]
        parts = [dict() for _ in range(len(splits))]

        for cellType in rme.cellTypes:
            for chrmState in rme.chromatinStates:
                label = f"{cellType}_{chrmState}"
                path = self.rmeBeds / f"{label}.bed"
                dataset = BedDataset(str(path), str(self.hg))

                # a set of mini batch lists, one for train/validation/test...
                clusterSets = makeClusters(list(iter(dataset)), clusterSize, splits)

                # iterate thrrough train/validation/test
                for i, mini in enumerate(clusterSets):
                    parts[i][label] = mini

        assert len(parts) == len(splits) + 1
        return {
            name: part for name, part in zip(["train", "validation", "test"], parts)
        }

    def _makeTriplets(
        self, clusterMap: dict[str, ClusterList], amnt: int
    ) -> Iterable[dict[str, Sequence[GenomicInterval]]]:
        axis1 = len(rme.cellTypes)
        axis2 = len(rme.chromatinStates)

        for _ in range(amnt):
            i1 = randint(0, axis1 - 1)
            j1 = randint(0, axis2 - 1)
            label1 = f"{rme.cellTypes[i1]}_{rme.chromatinStates[j1]}"

            i2 = (randint(0, axis1 - 2) + i1 + 1) % axis1
            j2 = (randint(0, axis2 - 2) + j1 + 1) % axis2
            label2 = f"{rme.cellTypes[i2]}_{rme.chromatinStates[j2]}"

            category1 = clusterMap[label1]
            anchor = random.choice(category1)
            positive = random.choice(category1)

            category2 = clusterMap[label2]
            negative = random.choice(category2)

            yield {"anchor": anchor, "positive": positive, "negative": negative}
