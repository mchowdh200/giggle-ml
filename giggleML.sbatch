#!/bin/bash

#SBATCH --partition=nvidia-a100
#SBATCH --nodelist=fijigpu-04
#SBATCH --gres=gpu:4
#SBATCH --mem=420G

#SBATCH --nodes=1
#SBATCH --ntasks=32

#SBATCH --job-name=giggleML
#SBATCH --output=giggleML-sbatch.out
#SBATCH --time=12:00:00

clear
echo Starting job.
source ~/rc/gpu.sh
source .venv/bin/activate
export PYTHONUNBUFFERED=1

time snakemake --cores all

echo Complete.
