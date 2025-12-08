"""
Completes a one-to-many query for scores over the roadmap epigenomics database.
"""

from pathlib import Path

import torch
import zarr
from tqdm import tqdm

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen.batch_infer import GenomicEmbedder
from giggleml.models.c_model import CModel
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.train.rme_caches import RmeBedCache, RmeFMCache
from giggleml.utils.parallel import dist_process_group
from giggleml.utils.torch_utils import get_rank, guess_device, launch_fabric


def fm_embed_one(bed_path: Path, embeds_out: Path, fasta: Path, batch_size: int = 70):
    with dist_process_group(), torch.inference_mode():
        device = guess_device()
        model = HyenaDNA("16k").eval().to(device)
        pipeline = GenomicEmbedder(model, batch_size)
        bed = BedDataset(bed_path, fasta)
        pipeline.to_disk([bed], [embeds_out])


def model_scores(
    cmodel_ckpt: Path,
    query_bed: Path,
    query_embeds: Path,
    rme_beds_dir: Path,
    rme_embeds_dir: Path,
    out_tsv: Path,
    batch_size: int,
):
    """produces a tsv file with header & columns (name, score)"""

    # retrieve embeddings for RME from cache
    bed_cache = RmeBedCache(rme_beds_dir)
    rme_bed_tensors = [bed_cache.get_tensor(x) for x in rme.bed_names]
    fm_cache = RmeFMCache(rme_embeds_dir)
    rme_embed_tensors = [fm_cache.get_tensor(x) for x in rme.bed_names]

    # retrieve query embeddings
    query_bed_tensor = RmeBedCache.clean_intervals(list(iter(BedDataset(query_bed))))
    query_embed_tensor = torch.from_numpy(zarr.open_array(query_embeds, mode="r")[:])

    interval_inputs = [query_bed_tensor] + rme_bed_tensors
    embed_inputs = [query_embed_tensor] + rme_embed_tensors
    assert len(interval_inputs) == len(embed_inputs)
    N = len(interval_inputs)

    with dist_process_group(), torch.inference_mode():
        fabric = launch_fabric()
        # load model
        model = CModel("16k", 512, 128, 2, 2)
        model = fabric.setup(model)
        model.mark_forward_method("distributed_embed")
        fabric.load(cmodel_ckpt, {"model": model})

        output = list()

        for i in tqdm(range(0, N, batch_size), "embedding"):
            j = min(i + batch_size, N)
            batch_in = (embed_inputs[i:j], interval_inputs[i:j])
            # rank-sharding happens internally
            batch_out = model.eval().distributed_embed(*batch_in).cpu()
            output.append(batch_out)
            fabric.barrier()

        cmodel_out = torch.cat(output)

    # pure-FM embeddings (mean pooling)
    query_embed_mean = query_embed_tensor.mean(dim=0)
    rme_embed_means = [x.mean(dim=0) for x in rme_embed_tensors]
    fm_out = torch.stack([query_embed_mean] + rme_embed_means)

    # scores
    cmodel_scores = (cmodel_out[0] - cmodel_out).norm(dim=1)
    fm_scores = (fm_out[0] - fm_out).norm(dim=1)

    # writing output

    if get_rank() != 0:
        return

    names = [query_bed.stem] + rme.bed_names

    with open(out_tsv, "w") as file:
        file.write("name\tcmodel_delta\tfoundation_model_delta\n")

        for name, cmodel_score, fm_score in tqdm(
            zip(names, cmodel_scores, fm_scores), desc="writing outputs"
        ):
            score_a = str(float(cmodel_score))
            score_b = str(float(fm_score))
            file.write(f"{name}\t{score_a}\t{score_b}\n")


if __name__ == "__main__":
    cmodel_ckpt = Path(
        "modelCkpts/cmodel_2025-12-07/ckpts_mix-mine/2025-12-07_21-23-10.pt"
    )
    query_bed = Path("data/Rheumatoid_arthritis/Rheumatoid_arthritis.bed")
    query_embeds = Path("data/Rheumatoid_arthritis/embeds.zarr")
    rme_beds_dir = Path("data/roadmap_epigenomics/beds")
    rme_embeds_dir = Path("data/roadmap_epigenomics/embeds")
    out_tsv = Path("data/Rheumatoid_arthritis/cmodel_scores.tsv")
    fasta = Path("data/hg/hg38.fa")

    # fm_embed_one(query_bed, query_embeds, fasta, batch_size=70)

    model_scores(
        cmodel_ckpt,
        query_bed,
        query_embeds,
        rme_beds_dir,
        rme_embeds_dir,
        out_tsv,
        batch_size=128,
    )
