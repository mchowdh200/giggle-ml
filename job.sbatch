#!/bin/bash

#SBATCH --partition=nvidia-a100
#   #SBATCH --nodelist=fijigpu-04
#SBATCH --gres=gpu:4
#SBATCH --mem=500G

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --job-name=giggleML
#SBATCH --output=jobOutput
#SBATCH --time=24:00:00

clear
echo Starting job.
source ~/rc/gpu.sh
source .venv/bin/activate
# mamba activate gml
export PYTHONUNBUFFERED=1

#data="/cache/siwa3657/"
data="data"
time python3 scripts/roadmapEpigenomicsEmbeds.py $data/roadmap_epigenomics $data/hg/hg19.fa
# time snakemake --cores 1 experiments/statTests/swtSynth.png

echo Complete.
