# GiggleML

Genomic similarity search based on deep set embeddings.

- Some newer results
  - [Sliding Window Test](./experiments/swt_combo.png)
  - [HyenaDNA UMAP](./experiments/roadmapEpigenomics/hyena_dna_umap.png)
- Some before May 2025 results
  - [Benchmark against legacy Giggle (with some interval transformations to increase performance)](./experiments/giggleBench/all-thirds/recallByGEOverlap.png)
  - [Results of Sheffield's statistical tests](./experiments/embed_analysis/sheffieldTestResults.md)
- [Statistical test methods for gauging embedding quality](wiki/sheffieldEmbedQualityTests.md)

## Install

**Install**

```bash
# 1. get the repo
git clone https://github.com/mchowdh200/giggle-ml.git
cd giggle-ml
# 2. build venv
uv sync
uv pip install -e .  # local self-install
```

**Other Dependencies**

If you are running a benchmark,

1. [Legacy Giggle](https://github.com/ryanlayer/giggle) if you intend to perform a benchmark
   against legacy Giggle. Acquiring results for comparison will require running
   legacy Giggle.
2. [Seqpare](https://github.com/deepstanding/seqpare)

## Datasets

Information about downloading and preprocessing datasets is given in [./scripts/datasets/]

Human genome (hg38.fa.gz) can be acquired [here](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/).

## Workflows

This system is designed to be used as a library and lacks complete CLIs.

### Produce Genomic Embeddings

To produce embeddings for a single `.bed` file with HyenaDNA,

```py

from pathlib import Path

from giggleml.analysis.rme_model_score import fm_embed_one

if __name__ == "__main__":
    query_bed = Path('query.bed')
    fasta = Path('hg38.fa')
    output = Path('embeds.zarr')
    fm_embed_one(query_bed, query_embeds, fasta, batch_size=70)

```

More generally for many files and other supported models,

```py

from pathlib import Path

import torch

from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.embed_gen.batch_infer import GenomicEmbedder
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.parallel import dist_process_group
from giggleml.utils.torch_utils import guess_device

if __name__ == "__main__":  # important
    bed_files = [ Path('bed1.bed'), Path('bed2.bed') ]
    outputs = [ Path('output1.zarr'), Path('output2.zarr') ]
    fasta = Path("hg38.fa")

    dataset = [BedDataset(path, fasta) for path in bed_files]

    with dist_process_group(), torch.inference_mode():
        model = HyenaDNA("16k").to(guess_device())
        pipeline = GenomicEmbedder(model, batch_size=256)
        pipeline.to_disk(dataset, outputs)
```

### Train CModel

These are the hyperparameters used to train CModel in the paper, except `mining_strategy="all"` was used during training steps 0-3499.

**Prepare dataset**

1. Note that this requires already preparred embeddings for roadmap epigenomics. Refer to previous workflow before proceeding.
2. Seqpare rank files. Refer to [./scripts/datasets/seqpare_ranks.md]

**Start training**

```py
from pathlib import Path

import torch.distributed as dist

from giggleml.models.c_model import CModel
from giggleml.train.train_orchestrator import Finetuner, TrainConfig
from giggleml.utils.torch_utils import launch_fabric


if __name__ == "__main__":
    launch_fabric()

    conf = TrainConfig(
        "train",
        base_model_dir=Path("modelCkpts/model_2025-12-07"),
        sprint_steps=7000,
        max_batches=7000,
        validation_freq=100,
        model=CModel("16k", 512, 128, 2, 2),
        mining_strategy="semi-hard",
        margin=0.01,
        learning_rate=2e-4,
        pk_ratio=0.5,
        positive_threshold=0.96,
        batch_size=128,
        sampling_rate=0.95,
    )

    ft = Finetuner(conf)
    ft.setup()
    ft.run()
```

### Roadmap Epigenomics Query

Perform a one-to-many query for a query.bed against the roadmap epigenomics database.

Note that this requires already preparred embeddings for the query file and roadmap epigenomics. Refer to other workflows on how to produce these before proceeding.

```py
from pathlib import Path

from giggleml.analysis.rme_model_score import model_scores

if __name__ == "__main__":
    cmodel_ckpt = Path("cmodel/2025-12-07_21-23-10.pt")
    query_bed = Path("Rheumatoid_arthritis.bed")
    query_embeds = Path("Rheumatoid_arthritis_embeds.zarr")
    rme_beds_dir = Path("roadmap_epigenomics/beds")
    rme_embeds_dir = Path("roadmap_epigenomics/embeds")
    out_tsv = Path("scores.tsv")
    fasta = Path("hg38.fa")

    model_scores(
        cmodel_ckpt,
        query_bed,
        query_embeds,
        rme_beds_dir,
        rme_embeds_dir,
        out_tsv,
        batch_size=128,
    )
```

This will produce a `.tsv` file, for example, the first 3 rows:

```tsv
name    cmodel_delta    foundation_model_delta
Rheumatoid_arthritis    0.0     0.0
Adult_Liver_Active_TSS  0.17578125      6.7265625
Adult_Liver_Flanking_Active_TSS 0.1806640625    6.65625
```

### Roadmap Epigenomics Ranks Analysis

Analyze & benchmark rankings of roadmap epigenomics items.

_Note:_ score `.tsv` files are required.

- refer to previous workflows to produce scores based on embedding models.
- refer to [seqpare_ranks.md](./scripts/datasets/seqpare_ranks.md) on building seqpare scores.
- refer to [Giggle](https://github.com/ryanlayer/giggle) on building giggle scores.

Build a Kendall's tau pairwise comparison matrix:

```py
from pathlib import Path

from giggleml.analysis.rme_rank_analysis import parse_scores_tsv, invert_scores, scores_to_ranks, kendall_tau_matrix

if __name__ == "__main__":
    data = Path("data/Rheumatoid_arthritis")

    # 1. load similarity scores

    # 0, 7, 2 corresponds to name col 0, data col 7, and 2 file extensions to strip from the names
    giggle_scores = parse_scores_tsv(data / "giggle_scores.tsv", 0, 7, 2)
    seqpare_scores = invert_scores(
        parse_scores_tsv(data / "seqpare_scores.tsv", 5, 4, 1)
    )
    cmodel_scores = invert_scores(parse_scores_tsv(data / "cmodel_scores.tsv", 0, 1))
    fm_scores = invert_scores(parse_scores_tsv(data / "cmodel_scores.tsv", 0, 2))

    # 2. load rankings

    giggle_ranks = scores_to_ranks(giggle_scores)
    seqpare_ranks = scores_to_ranks(seqpare_scores)
    cmodel_ranks = scores_to_ranks(cmodel_scores)
    fm_ranks = scores_to_ranks(fm_scores)

    # 3. plot

    names = ["CModel", "HyenaDNA", "Seqpare", "Giggle"]
    ranks = [cmodel_ranks, fm_ranks, seqpare_ranks, giggle_ranks]
    fig = kendall_tau_matrix(zip(names, ranks))
    fig.savefig(data / "rme_kt_matrix.png"))
```
