from collections.abc import Iterable

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import giggleml.dataWrangling.fasta as fasta
from giggleml.embedGen.embedModel import HyenaDNA
from giggleml.train import roadEpGroups
from giggleml.train.roadEpTripletFT import Cluster, LitModule


class Collate:
    def __init__(self, hg: str):
        self.hg: str = hg

    def make(self, batch) -> Iterable[Cluster]:
        for bedFile in batch:
            for cluster in bedFile:
                yield fasta.map(cluster, self.hg)

    def __call__(self, batch) -> list[Cluster]:
        return list(self.make(batch))


if __name__ == "__main__":
    # INFO: config

    model = HyenaDNA("32k", training=True)
    clusterSize = 4
    clusterSize = 2
    clustersPerLabel = 16
    clustersPerLabel = 2
    labelsPerBatch = clustersPerLabel * 4
    lr = 0.00002
    betas = (0.9, 0.999)
    margin = 1
    maxEpochs = 5

    embedBatchSize = 64
    subWorkers = 4
    subWorkers = 0
    workers = torch.cuda.device_count() or 1

    ckptPath = "models/hyenaFT-32k-061625"
    rmeBeds = "data/roadmap_epigenomics/beds"
    hg = "data/hg/hg19.fa"

    # INFO: data

    print("Training", ckptPath)
    datasets = roadEpGroups.makeDatasets(
        rmeBeds, hg, clusterSize, clustersPerLabel, 0.8, 0.5
    )
    print("Datasets preparred.")
    trainDataset = datasets["train"]
    testDataset = datasets["test"]
    validationDataset = datasets["validation"]

    trainLoader = DataLoader(
        trainDataset,
        batch_size=labelsPerBatch,
        num_workers=subWorkers,
        persistent_workers=subWorkers > 0,
        collate_fn=Collate(hg),
    )

    testLoader = DataLoader(
        testDataset,
        batch_size=labelsPerBatch,
        num_workers=subWorkers,
        persistent_workers=subWorkers > 0,
        collate_fn=Collate(hg),
    )

    # INFO: train

    print("Starting...")
    module = LitModule(
        model,
        embedBatchSize,
        clusterSize,
        labelsPerBatch,
        clustersPerLabel,
        workers,
        lr,
        betas,
        margin,
    )
    trainer = pl.Trainer(
        default_root_dir=ckptPath,
        max_epochs=maxEpochs,
        # accelerator="gpu", # default: auto
        # strategy="ddp", # default: auto
        devices=workers,
        # FIXME: rm
        fast_dev_run=3,
    )

    # --- Start Training ---
    trainer.fit(module, trainLoader, testLoader)
