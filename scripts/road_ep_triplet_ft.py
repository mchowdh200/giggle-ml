from collections.abc import Iterable

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import giggleml.data_wrangling.fasta as fasta
from giggleml.embed_gen.embed_model import HyenaDNA
from giggleml.train import road_ep_groups
from giggleml.train.road_ep_triplet_ft import Cluster, LitModule


class Collate:
    def __init__(self, hg: str):
        self.hg: str = hg

    def make(self, batch) -> Iterable[Cluster]:
        for bed_file in batch:
            for cluster in bed_file:
                yield fasta.map(cluster, self.hg)

    def __call__(self, batch) -> list[Cluster]:
        return list(self.make(batch))


if __name__ == "__main__":
    # INFO: config

    model = HyenaDNA("32k", training=True)
    cluster_size = 4
    cluster_size = 2
    clusters_per_label = 16
    clusters_per_label = 2
    labels_per_batch = clusters_per_label * 4
    lr = 0.00002
    betas = (0.9, 0.999)
    margin = 1
    max_epochs = 5

    embed_batch_size = 64
    sub_workers = 4
    sub_workers = 0
    workers = torch.cuda.device_count() or 1

    ckpt_path = "models/hyenaFT-32k-061625"
    rme_beds = "data/roadmap_epigenomics/beds"
    hg = "data/hg/hg19.fa"

    # INFO: data

    print("Training", ckpt_path)
    datasets = road_ep_groups.make_datasets(
        rme_beds, hg, cluster_size, clusters_per_label, 0.8, 0.5
    )
    print("Datasets preparred.")
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    validation_dataset = datasets["validation"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=labels_per_batch,
        num_workers=sub_workers,
        persistent_workers=sub_workers > 0,
        collate_fn=Collate(hg),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=labels_per_batch,
        num_workers=sub_workers,
        persistent_workers=sub_workers > 0,
        collate_fn=Collate(hg),
    )

    # INFO: train

    print("Starting...")
    module = LitModule(
        model,
        embed_batch_size,
        cluster_size,
        labels_per_batch,
        clusters_per_label,
        workers,
        lr,
        betas,
        margin,
    )
    trainer = pl.Trainer(
        default_root_dir=ckpt_path,
        max_epochs=max_epochs,
        # accelerator="gpu", # default: auto
        # strategy="ddp", # default: auto
        devices=workers,
        # FIXME: rm
        fast_dev_run=3,
    )

    # --- Start Training ---
    trainer.fit(module, train_loader, test_loader)
