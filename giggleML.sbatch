#!/bin/bash

#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:2
#SBATCH --mem=160G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --job-name=giggleML
#SBATCH --output=giggleML-sbatch.out

# #SBATCH --time=00:05:00

clear

# because apparently srun fights my permission configuration
on_exit() {
  chmod +x ./run.sh
}

trap on_exit EXIT

sgpu "
echo \"Starting job.\"

source $HOME/rc/gpu.sh
conda deactivate
conda activate gml

python src/embedSets.py

conda deactivate
"

echo "Complete."