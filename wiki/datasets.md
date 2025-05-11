# roadmap_epigenomics

Is based on ChromHMM segmentation data from the NIH, [(1)](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html). The data was then processed, split and sorted in some way, by Giggle, [(2)](https://github.com/ryanlayer/giggle/blob/aeda9930454fe9a06383c8cff315caf56af11b49/examples/rme/README.md?plain=1#L25). Giggle provides this data (post-processed) [here](https://s3.amazonaws.com/layerlab/giggle/roadmap/roadmap_sort.tar.gz) which is what this library refers to as the "Roadmap Epigenomics" dataset.

According to **(1)**, "The functional annotations used were as follows (All coordinates were relative to the hg19 version of the human genome)". So, this library maps the intervals to sequences using `hg19`.

```py
--- Statistics Summary ---
Files successfully processed: 1905
Total valid intervals found: 55605005

# Interval Length Stats (across all intervals):
  Min Length:    200
  Max Length:    30384000
  Mean Length:   7064.05
  Length Stdev: 92692.10

# Intervals Per File Stats:
  Min Intervals/File:    5
  Max Intervals/File:    146238
  Mean Intervals/File:   29188.98
  Stdev Intervals/File: 29804.55
------------------------```
(from `src/scripts/bedStats.py`)

After chunking intervals to max length of 32,768
there are 60,821,687 intervals.
~10% increase.

# ENCFF478KET_small

Based on [ENCFF478KET](https://www.encodeproject.org/files/ENCFF478KET/) -- pipelines:
- Transcription factor ChIP-seq 2,
- Histone ChIP-seq 2 (unreplicated),
- Histone ChIP-seq 2,
- Transcription factor ChIP-seq 2 (unreplicated)

It is to be used with GRCh38.

`sample.bed` and `query.bed` were produced from `ENCFF478KET.bam`.
`sample.bed`: 100,000 intervals were sampled after filtering for chr1
`query.bed`: First 1,000 intervals of `sample.bed`

