#!/bin/bash

cd "$(dirname "$0")"
data=../data

if [ ! -d $data ]; then
  echo "Expected a folder at project root, 'data'"
  exit -1
fi

cd $data

# ----------------------------------------------
#     Roadmap Epigenomics
# ----------------------------------------------

if [ -d roadmap_epigenomics ]; then
  echo "Already have roadmap_epigenomics"
else
  read -p "Get roadmap_epigenomics? (Y/n): " confirm

  if [[ "$confirm" == [yY] || "$confirm" == "" ]]; then
    wget https://s3.amazonaws.com/layerlab/giggle/roadmap/roadmap_sort.tar.gz
    tar -zxvf roadmap_sort.tar.gz
    rm roadmap_sort.tar.gz

    if [ ! -d roadmap_epigenomics/beds ]; then
      mkdir -p roadmap_epigenomics/beds
    fi

    mv roadmap_sort/* roadmap_epigenomics/beds
    rmdir roadmap_sort
    echo "For performance, you should gunzip data/roadmap_epigenomics/beds/*.gz"
  fi
fi

# ----------------------------------------------
#     Small Roadmap Epigenomics
# ----------------------------------------------

if [ -d rme_small ]; then
  echo "Already have rme_small"
else
  read -p "Get rme_small? (Y/n): " confirm

  if [[ "$confirm" == [yY] || "$confirm" == "" ]]; then
    # assuming roadmap_epigenomics is already prepared

    if [ ! -d rme_small/beds ]; then
      mkdir -p rme_small/beds
    fi

    rme=roadmap_epigenomics/beds
    ls $rme | head -n 56 | xargs -I{} cp $rme/{} rme_small/beds
    cd rme_small/beds

    if (ls *.gz > /dev/null 2>&1); then
      gunzip *.gz;
    fi

    beds=$(ls *bed)
    echo "$beds" | xargs -I{} bash -c "head -n 1000 {} > {}_"
    echo "$beds" | xargs -I{} mv {}_ {}
    cd ..
  fi
fi

# ----------------------------------------------
# TODO:
#     ENCFF478KET_small
# ----------------------------------------------

