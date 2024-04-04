# Embed System

A work in progress..

Currently taking embeds from the genomic benchmark dataset and generating
embeddings on gpus in parallel.

## Building

1. `pip3 install -r requirements.txt`
2. You may need `gcc` and `python3-dev` to build biopython as well as `git-lfs`.

- `apt-get install gcc python3-dev git-lfs`

### Run

1. `python3 inference_batch.py`
