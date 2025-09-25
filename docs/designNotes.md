## Seqpare

- Self-consistent metric. Due to the triangle inequality holding, I can make assumptions about the relative similarity between A and C where A and C are both independently close to B. This lets me trivially sample clusters seqpare-based KNN results.
- Triangle inequality only holds when converting seqpare into a distance function (1-SeqpareSimilarity).

## Triplet Loss vs Contrastive Loss

Both are applicable to learn the metric.

- Expensive: training on embed **sets**, with a foundation model (hyena DNA) that embeds individual elements. To compute the loss, we have embed repeatedly on a subsample to approximate the mean embeddings of the set.
- Mining triplets within the batch allows us to reuse set embeddings (expensive part).
- Triplet loss theoretical benefits. Due to all items in the batch being mutually related, the loss more directly reflects the intended structure of the embedding space.
- Online, hard triplet mining focuses on important examples. Efficiently, because all set embeddings are used for at least one triplet.

## Tiling Algorithm

3.2e9 base pairs vs ~3.9e11 for roadmap epigenomics

- Memoization, to provide huge speed up.
- Speed is important: will allow embedding all of UCSC?

- Algorithm, 9/23/2025
  - Tiles at increasing sizes (base size)\*2^K for the K-th layer. Each layer has a few offsets that linearly divide the tile spacing for that layer.
  - There are (layers \* offsets per layer) full-genome tilings.
  - Interval -> Tile Composition Algorithm: greedy, picks the largest most centered tile within the interval first. Recursively tiles the remainder. Refuses to yield a tile that adds more "noise" than coverage.

## Other

- Permutation invariant. Giggle, the inspiration, is permutation invariant to intervals. As is seqpare. And, a permutation invariant embedding architecture.
- Naive set level triplet mining is too expensive. Already exceeding A100 80GB memory limits with a batch size of 16 and 20 intervals per item. 20 intervals is questionable because CLT kicks in at 40 as a rule of thumb. And 16 is very small for proper triplet mining. Additionally, maxing out GPU memory incurs a strong performance penalty because we have no reserve space for triplet mining over the overall combined batch between all GPUs. Changing strategies...
