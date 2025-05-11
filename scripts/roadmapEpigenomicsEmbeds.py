# WARN: This script does NOT enable interrupted job continuation.
#  Waiting on changes to pipeline first. Until then, this is only
#  an alternative to the (recommended) snakemake workflow.
"""
Produces embeddings for the RoadmapEpigenomics Dataset

Replaces the corresponding snakemake workflow. A separate script makes sense
because there can be interruptions and this script implicitly provides functionality
for job continuation.  Additionally, when the job is extremely large, it is better to
start/stop it manually rather than let snakemake prepare it flexibly.
"""

import argparse
import os
from os.path import basename

from giggleml.dataWrangling.intervalDataset import BedDataset
from giggleml.embedGen import meanEmbedDict
from giggleml.embedGen.embedModel import HyenaDNA, TrivialModel
from giggleml.embedGen.embedPipeline import EmbedPipeline


def build(roadmapDir: str, hg19: str):
    bedNames = os.listdir(roadmapDir + "/beds")
    names = list[str]()

    for path in bedNames:
        if path.endswith(".bed.gz"):
            names.append(basename(path)[:-7])
        elif path.endswith(".bed"):
            names.append(basename(path)[:-4])
        else:
            raise ValueError(f"Expecting .bed.gz or .bed files ({path})")

    beds = [f"{roadmapDir}/beds/{bed}" for bed in bedNames]
    outPaths = [f"{roadmapDir}/embeds/{name}.npy" for name in names]

    # INFO: inferene parameters
    batchSize = 256
    subWorkerCount = 4
    model = HyenaDNA("32k")
    # model = TrivialModel(32768)

    # big job
    inputData = [BedDataset(bed, associatedFastaPath=hg19) for bed in beds]
    EmbedPipeline(model, batchSize, subWorkers=subWorkerCount).embed(
        intervals=inputData, out=outPaths
    )

    # little job (produce $master)
    meanEmbedDict.build(outPaths, f"{roadmapDir}/embeds/master.pickle")
    print("Complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for roadmap epigenomics")
    parser.add_argument(
        "roadmapDir",
        help="Dir path; should include subdirectories beds/ and embeds/",
        default="data/roadmap_epigenomics/beds",
    )
    parser.add_argument("hg19", help="path to hg19.fa or hg19.fa.gz", default="data/hg/hg19.fa")

    args = parser.parse_args()
    roadDir = args.roadmapDir
    hg19 = args.hg19

    build(roadDir, hg19)


if __name__ == "__main__":
    main()
