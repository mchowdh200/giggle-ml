#!/bin/bash

#SBATCH --partition=nvidia-a100
#SBATCH --nodelist=fijigpu-06
#SBATCH --gres=gpu:2
#SBATCH --mem=420G

#SBATCH --nodes=1
#SBATCH --ntasks=32

#SBATCH --job-name=giggleML
#SBATCH --output=giggleML-sbatch.out
#SBATCH --time=9:00:00

clear
echo Starting job.
source ~/rc/gpu.sh
mamba activate gml

export PYTHONUNBUFFERED=1
time python -u src/embedSets.py
#python -m cProfile -o ./prof.pstats src/embedSets.py

mamba deactivate
echo Complete.