#!/bin/bash

# because apparently srun fights my permission configuration
on_exit() {
  chmod +x ./sbatch.sh
}

trap on_exit EXIT

clear
echo -e ">\n" > giggleML-sbatch.out
sbatch giggleML.sbatch
tail -f giggleML-sbatch.out