#!/bin/bash

#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:2
#SBATCH --mem=420G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --job-name=giggleML
#SBATCH --output=giggleML-sbatch.out
#SBATCH --time=9:00:00

clear

sgpu "
echo \"Starting job.\"

source $HOME/rc/gpu.sh
conda deactivate
conda activate gml

python src/embedSets.py

conda deactivate
"

echo "Complete."