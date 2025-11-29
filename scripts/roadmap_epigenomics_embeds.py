# WARN: This script does NOT enable interrupted job continuation.


from pathlib import Path

import torch

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen.batch_infer import GenomicEmbedder
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.parallel import dist_process_group
from giggleml.utils.torch_utils import guess_device


def main():
    rme_dir = Path("data/roadmap_epigenomics")
    bed_slice = rme.bed_names[300:1000]
    bed_paths = [rme_dir / "beds" / f"{name}.bed" for name in bed_slice]
    out_paths = [rme_dir / "embeds" / f"{name}.zarr" for name in bed_slice]
    fasta = "data/hg/hg38.fa"
    beds = [BedDataset(path, fasta) for path in bed_paths]
    batch_size = 256

    with dist_process_group(), torch.inference_mode():
        device = guess_device()
        model = HyenaDNA("16k").eval().to(device)
        pipeline = GenomicEmbedder(model, batch_size)
        pipeline.to_disk(beds, out_paths)


if __name__ == "__main__":
    main()
