from pathlib import Path

from giggleml.train.rme_clusters_dataset import RmeSeqpareClusters
from giggleml.train.seqpare_db import SeqpareDB


def main():
    data_path = Path("data", "roadmap_epigenomics")
    sdb = SeqpareDB(data_path / "seqpareRanks")
    threshold = 0.96
    dset = RmeSeqpareClusters(data_path / "beds", sdb, 1, 0, threshold)

    # print(
    #     [int(sum(sdb.fetch_mask(name, threshold))) for name in dset.allowed_rme_names]
    # )

    dset_iter = iter(dset)

    for i in range(1):
        item = next(dset_iter)
        print("good")


if __name__ == "__main__":
    main()
