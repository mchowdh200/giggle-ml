# Training paradigm

## Seqpare

- Self-consistent metric. Due to the triangle inequality holding, I can make assumptions about the relative similarity between A and C where A and C are both independently close to B. This lets me trivially sample clusters seqpare-based KNN results.
- Triangle inequality only holds when converting seqpare into a distance function (1-SeqpareSimilarity).

## Triplet Loss vs Contrastive Loss

Both are applicable to learn the metric.

- Expensive: training on embed **sets**, with a foundation model (hyena DNA) that embeds individual elements. To compute the loss, we have embed repeatedly on a subsample to approximate the mean embeddings of the set.
- Mining triplets within the batch allows us to reuse set embeddings (expensive part).
- Triplet loss theoretical benefits. Due to all items in the batch being mutually related, the loss more directly reflects the intended structure of the embedding space.
- Online, hard triplet mining focuses on important examples. Efficiently, because all set embeddings are used for at least one triplet.

## Similarity graph

Hard triplets are normally mined by selecting, for an anchor, a positive element that the embedding model thinks is particularly different. But my data consists only of elements and a continuous similarity value between elements. If elements are sufficiently similar, we call it positive. So, it's a graph. I can't sample sufficiently distant anchors in the graph for which to draw clusters around those anchors, because the clusters would have overlap. If clusters have overlap, I must be sure not to define elements in distinct neighborhoods as negative. The solution obvious, but trickier. The mining batch is not a set of distinct clusters, it is series of sub-graphs where any node can be anchor so long as we determine its unique positive neighborhood.

## C Model

Why?

- The system operates at set level; permutation invariant. Giggle, the inspiration, is permutation invariant to intervals. As is seqpare. And, a permutation invariant embedding architecture.
- Naive set level triplet mining is too expensive. Already exceeding A100 80GB memory limits with a batch size of 16 and 20 intervals per item. 20 intervals is questionable because CLT kicks in at 40 as a rule of thumb. And 16 is very small for proper triplet mining. Additionally, maxing out GPU memory incurs a strong performance penalty because we have no reserve space for triplet mining over the combined batch between all GPUs.

Deep sets architecture on HyenaDNA embeddings  
`interval set --fasta--> sequence set --HyenaDNA--> sequence embeddings --C Model--> (a single) set embedding`

HyenaDNA sequence embeddings across all sets in the batch are passed through $\phi$. We use gradient checkpointing over the $\phi$ MLP -- activations are thrown out, but inputs (into the MLP) cached. The loss can be calculated with a low memory footprint using this technique. The backward pass is efficient because gradient accumulation occurs automatically by pytorch. A running average over the $\phi$ gradients; no entire set of activations in memory at once. It's effectively no limit on the size of sets we can process because we've pushed the memory complexity into the time complexity and made the model lightweight.

# Embedding pipeline

I've identified that pipeline designs are hard to generalize because the accelerator and context varies through the pipeline. Boundaries, obvious in one layer, may need to become data in another. Subsequent boundaries don't have to match the rest. Order isn't guaranteed, but often required. Some items can become multiple. I have managed to generalize all my usages.

```
    [ N items stream ] -> [ M > N items ] -> [ M//b batches ] -> [ M//b embeds ] -> [ N outputs ]
                 ("preprocess")   (DataLoader batching) | (model call)  |  ("postprocess")
                                                        |collate        |decollate
```

Additionally tokenization is often dependent on the type of model and input data type. To handle that, I take in a collate_fn and decollate_fn as parameters used just before and after the model call. In a situation where genomic intervals are provided and chunked, tokenization can be used conditionally with fasta-mapping simply by passing the proper functions.  
This allows a stream of genomic interval streams to be processed by flattening and zipping with indices. The DataLoader will create batch boundaries invariant of set boundaries or the growth-factor due to chunking. This guarantees stable batching and maximum throughput entirely invariant to the input shape or worker count.

The DataLoader sub-workers and top-level workers all work out of sync and so output order is generally far from guaranteed and difficult to repair in a streaming way. We can get eventual ordering by tagging inputs with indices used to direct outputs. We can use Zarr arrays (which support) concurrent writing to different chunks to collect sorted outputs without a non-linear cost in the same way that we can linearly sort a distinct and fixed length list of numbers. Disk IO is free because we have essentially three computational devices: CPU (top-level worker), CPU (sub-worker), GPU(s). The sub-workers continuously prepare batches for the GPU to continuously embed for the top-level workers to continuously write. The overlapping nature of organization amortizes transit costs.

The pipeline could be theoretically used to process an infinite data stream so long as resumption were accounted for.
Breaking the pipeline into pre/postprocess, model, de/collate steps is "safe" because it prevents the user from operating directly on the infinite data stream.

# Other

## Tile Index

3.2e9 base pairs vs ~3.9e11 for roadmap epigenomics

- Memoization, to provide huge speed up.
- Speed is important: will allow embedding all of UCSC?

### Tiling Algorithm

- Algorithm, 9/23/2025
  - Tiles at increasing sizes (base size)\*2^K for the K-th layer. Each layer has a few offsets that linearly divide the tile spacing for that layer.
  - There are (layers \* offsets per layer) full-genome tilings.
  - Interval -> Tile Composition Algorithm: greedy, picks the largest most centered tile within the interval first. Recursively tiles the remainder. Refuses to yield a tile that adds more "noise" than coverage.

### Death

No longer viable :(

Recent experiments have revealed huge embedding error due to scale change. The hierarchical tiling strategy was previously thought to incur a 15% average error after hyperparameter adjustment, but because biological intervals are generally somewhat similar at a distance, it's effectively a 40% error -- destroyed the information. No set of hyperparameters could reduce this. The positional offset trick would likely dramatically reduce error because embedding error due to translation is very low for HyenaDNA. But dropping the hierarchy in tile composition would mean far more tiling sizes to encompass bed file variance and given the genome size, this is not feasible. For 3B 1k-tiles, we can only feasibly tile that about 200 times (10x size of rme). Including positional offsets, it's far too expensive.

# Untested experimental variables

- foundation model things:
  - best foundation model type?
  - do foundation models with large context do worse on small sequences?
  - where do errors in the sequences matter most wrt embedding error?
    - especially +/- nucleotides at start/end
  - is mean aggregation of sequence chunks naive? (it assumes chunks are positionally invariant)
    - should we use a baby deep-sets architecture to stich chunk embeds? Be wary of [vanishing gradients](https://pmc.ncbi.nlm.nih.gov/articles/PMC10465016/#:~:text=At%20high%20depths%2C%20Deep%20Sets,see%20Section%206%20for%20details)
- long-range proximity and embedding distance relationship?
- how well do we generalize to unseen bed files?
  - what about bed files "between" roadmap epigenomics bed files?
- is it important to train tiling-aware?
- bed files are in fact not orderless....

# Other random things things I did

- a couple (fasta-mapped) bed file entropy measures
- embedding sheffield tests & SWT
- verifying the CLT applies to bed files...
- umapped hdna on rme
- all those direct interval-interval direct benchmarks against giggle
  - testing how interval alteration introduces error
