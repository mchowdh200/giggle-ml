# GiggleML

Giggle is a genomic search engine. Internally, it finds intervals
that overlap. GiggleML extends to sequence information using ML
embeddings and uses a vector database to scale.

More info

1. [Current state of the project](wiki/state.md)
2. [Statistical test methods for gauging embedding quality](wiki/sheffieldEmbedQualityTests.md)
3. A few results
   - [Results of  Sheffield's statistical tests](./embedAnalysis/sheffieldTestResults.md)
   - [Sliding Window Test](./embedAnalysis/swt.png)
   - [Benchmark against legacy Giggle (with some interval transformations to increase performance)](./experiments/giggleBench/all-thirds/recallByGEOverlap.png)
   - [HyenaDNA UMAP](./embedAnalysis/umap/umap.png)

## Usage

### Prerequisites

Windows is not supported.

1. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)
3. [Legacy Giggle](https://github.com/ryanlayer/giggle) dependencies $\to$ **IF** you intend to perform a benchmark
   against legacy Giggle. Acquiring results for comparison will require running
   legacy Giggle.
    - More information about installing and querying with legacy Giggle can be found on its README.
4. Interval data and a reference genome
    - I recommend the Roadmap Epigenomic dataset featured in the legacy Giggle README.

---

1. Initialize the conda environment:
   `conda env create -f conda/gml.yml`
2. Proceed to a specific workflow...

## Workflows

Prior to running any workflow, make sure to configure the parameters first.
Workflow parameters are included in snakemake files and in the [./config.yaml](./config.yaml).

### Large scale embedding generation (Roadmap Epigenomics embeddings)

Embeds a large series of bed files.

`snakemake RoadmapEpigenomicsEmbeddings`  
Note that this rule can be configured to embed any collection of interval sets
(not just roadmap epigenomics).

### Embedding quality tests

`snakemake embedQualityTests`

This executes the 4 [statistical tests](./wiki/sheffieldEmbedQualityTests.md) and SWT.

### Legacy Giggle benchmark

1. Prepare the datasets  
This assumes two interval datasets, `query` and `sample`. The query dataset is used
to perform queries (searches) on the samples. The `sample` dataset should be much larger
than the `query` dataset.
2. Execute legacy Giggle to create baseline results. The Giggle index should be created
on `sample` and searched using `query`.
You will need to use the `-v -o` flags during search.


To benchmark against all interval transformation experiments,  
`snakemake giggleBenchmark_*` or  
`snakemake giggleBenchmark_straight` for no transformation.

### Sequence synthesis

`snakemake synthesizeSequences`

The system is capable of generating multiple bed files at once by increasing
the amount of `outFiles`. It will always create a single fasta file.

### Transformed interval embeddings

To run all interval transformation experiments,  
`snakemake embedsAfterTransform_*`

Transformed interval experiments were done primarily for the giggle benchmark.  
Embedding without transformation is possible with
`snakemake embedsAfterTransform_straight`.
