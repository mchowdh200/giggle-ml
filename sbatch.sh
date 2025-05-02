#!/bin/bash

# because apparently srun fights my permission configuration
on_exit() {
  chmod +x ./sbatch.sh
}

trap on_exit EXIT

clear
echo -e "..." > jobOutput
sbatch job.sbatch
tail -f jobOutput
