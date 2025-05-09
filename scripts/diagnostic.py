import numpy as np

from giggleml.dataWrangling import fasta
from giggleml.dataWrangling.intervalDataset import LateIntervalDataset, MemoryIntervalDataset
from giggleml.embedGen.embedModel import HyenaDNA
from giggleml.embedGen.gpuMaster import GpuMaster
from giggleml.utils.types import GenomicInterval


def main():
    problemInterval: GenomicInterval = ("chr1", 99893733, 99893809)
    chrm, start, end = problemInterval
    size = end - start
    print("size:", size)

    dataset = MemoryIntervalDataset([problemInterval], "data/hg/hg38.fa")
    seq = fasta.map(dataset)[0]
    print("seq:", seq)

    model = HyenaDNA("1k")
    embedRaw = model.embed([seq])
    print("raw embed:", embedRaw)

    GpuMaster(model, 1, 1, 0).batch([dataset], ["scripts/diagnostic.npy"])
    embed = np.memmap("scripts/diagnostic.npy", np.float32, "r", 0, (1, model.embedDim))
    print("pipeline embed:", embed)


if __name__ == "__main__":
    main()
