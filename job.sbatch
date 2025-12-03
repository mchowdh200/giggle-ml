#!/bin/bash

#SBATCH --partition=nvidia-a100
#  #SBATCH --nodelist=fijigpu-06
#SBATCH --gres=gpu:2
#SBATCH --mem=500G

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --job-name=giggleML
#SBATCH --output=jobOutput
#SBATCH --time=0-01:00:00

clear
echo Starting job.
source ~/rc/gpu.sh
export PYTHONUNBUFFERED=1

# uv sync
# uv pip install -e .
mamba activate gml3

# data="/cache/siwa3657/data"
data="data"
# time python3 scripts/roadmapEpigenomicsEmbeds.py $data/roadmap_epigenomics $data/hg/hg19.fa
# time snakemake --cores 1 all
time py-spy record -o data/spy.svg -r 20 --subprocesses -- torchrun --nproc_per_node=2 scripts/hparam_search.py

echo Complete.
