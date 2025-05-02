#!/bin/bash

#SBATCH --partition=nvidia-a100
#SBATCH --nodelist=fijigpu-04
#SBATCH --gres=gpu:4
#SBATCH --mem=420G

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
export PYTHONUNBUFFERED=1

time python3 scripts/roadmapEpigenomicsEmbeds.py /cache/siwa3657/data/roadmap_epigenomics /cache/siwa3657/data/hg/hg19.fa

echo Complete.
