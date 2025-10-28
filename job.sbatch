#!/bin/bash

#SBATCH --partition=nvidia-a100
#  #SBATCH --nodelist=fijigpu-06
#SBATCH --gres=gpu:4
#SBATCH --mem=500G

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --job-name=giggleML
#SBATCH --output=jobOutput
#SBATCH --time=1-00:00:00

clear
echo Starting job.
source ~/rc/gpu.sh
export PYTHONUNBUFFERED=1

uv sync
uv pip install -e .

# data="/cache/siwa3657/data"
data="data"
# time python3 scripts/roadmapEpigenomicsEmbeds.py $data/roadmap_epigenomics $data/hg/hg19.fa
# time snakemake --cores 1 all
time uv run torchrun --nproc_per_node=4 scripts/hparam_search.py

echo Complete.
