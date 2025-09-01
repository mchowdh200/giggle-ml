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

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen import mean_embed_dict
from giggleml.embed_gen.embed_model import HyenaDNA
from giggleml.embed_gen.embed_pipeline import DirectPipeline


def build(roadmap_dir: str, hg19: str):
    bed_names = os.listdir(roadmap_dir + "/beds")
    names = list[str]()

    for path in bed_names:
        if path.endswith(".bed.gz"):
            names.append(basename(path)[:-7])
        elif path.endswith(".bed"):
            names.append(basename(path)[:-4])
        else:
            raise ValueError(f"Expecting .bed.gz or .bed files ({path})")

    beds = [f"{roadmap_dir}/beds/{bed}" for bed in bed_names]
    out_paths = [f"{roadmap_dir}/embeds/{name}.npy" for name in names]

    # INFO: inferene parameters
    batch_size = 64
    sub_worker_count = 4
    model = HyenaDNA("32k")
    # model = TrivialModel(32768)

    # big job
    sr = 0.1

    if sr != 1:
        print(f"Applying a {sr*100}% subsampling rate")

    input_data = [BedDataset(bed, associated_fasta_path=hg19, sampling_rate=sr) for bed in beds]
    DirectPipeline(model, batch_size, sub_workers=sub_worker_count).embed(
        intervals=input_data, out=out_paths
    )

    # little job (produce $master)
    mean_embed_dict.build(out_paths, f"{roadmap_dir}/embeds/master.pickle")
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
    road_dir = args.roadmap_dir
    hg19 = args.hg19

    build(road_dir, hg19)


if __name__ == "__main__":
    main()
