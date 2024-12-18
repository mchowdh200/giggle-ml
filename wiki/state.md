Simon Walker  
Dec 18th, 2024

# State of the Project

## Complete

### Embedding generation framework

- Batch inference implemented & tested at a scale of 50+ million data points.
    - Required parallelization solutions to support efficiency across multiple A100s, large inputs, and networked
      resources.
- The inference framework is modularized to be agnostic to the ML model.
    - Support for hyenaDNA; implemented a very small wrapper. A single vector is produced from the hyenaDNA transformer
    - via mean aggregation.
    - Genomic Benchmark dataset integration.
    - Support for "Region2Vec": a model by Sheffield et al. It was designed for a similar purpose to GiggleML but does
      not operate on nucleotide information.

### Embedding diagnostic

- Brief literature review mostly regarding papers by Sheffield et al. In particular, statistical tests for embedding quality are summarized the methods in a one-pager, [here](sheffieldEmbedQualityTests.md).
- Statistical tests reimplemented.
    - They were originally implemented by Chris (Ex-LayerLab), but for smoother system-wide integration, among other
      issues, I reimplemented all of these tests. The 4 tests are [summarized](sheffieldEmbedQualityTests.md).
- Sliding Window Test (SWT) implemented. This is another diagnostic similar to Sheffield's tests. It regards the
  embedding similarity with interval overlap relationship; relevant to achieving capabilities of the legacy Giggle
  system.
- Implemented two approaches to computing the entropy of an interval set. The idea was to determine if a theme could be
  quantitatively verified in bed files with a specific function. Analysis of these results compared to the other metrics
  is not yet complete.
- Interval synthesizer to verify embedding diagnostics performance on random data.
    - Interval sizes are controlled and are placed realistically, within chromosome boundaries. Chromosome sizes were
      accounted for. A minimal fasta file with corresponding sequence information is also generated.

### Legacy Giggle Benchmark

This includes not only the benchmark but a series of experiments conducted with the intention to increase recall (wrt the
legacy system) without significant performance burden.

This required
- Implementation of the new core algorithm involving vector database population.
- Plotting recall with regard to various conditions including tolerance for poorly overlapping intervals.

Recall suffered at first but was ultimately increased dramatically by modifying intervals before being embedded
and indexed by the vector database. There were 12 variations of interval transformations explored,
but they are generally...

Interval Transformations
- Chunk
- Grow
- Shrink
- Translate

### Additionally
- UMAPed some hyenaDNA embeddings.
- Briefly explored the capacity of the system to find the original embeddings after mean aggregation.

And,  
Maintained fiji, super cluster, support.  
A major time sync.

## Codebase

- Documented.
- Reproducible (among other things, random number generator seeds are version-controlled and documented).
- Important entry points have associated snakemake rules.
- Partial test coverage.
