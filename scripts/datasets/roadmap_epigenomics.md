# roadmap_epigenomics

Roadmap epigenomics chmm marks are provided as a series of 127 bed files named by edacc IDs where each region is annotated with one of fifteen chromatin states. To make the data easier to work with, we: split, sort and rename files into the form `{cellType}_{chromatinState}.bed`.

```sh
# get the raw data
wget http://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/all_hg38lift.mnemonics.bedFiles.tgz -O raw.tgz
tar -xzvf raw.tgz

# split and rename
#   127 cell types by 15 chromatin states gives us 1,905 files with 56,440,237 intervals.
out="data/roadmap_epigenomics/beds" # or wherever you like
python rme_split.py "*.bed.gz" $out

# sort with bedtools
ls $out/*.bed | xargs -I {} sh -c "bedtools sort -i {} > {}_ && mv {}_ {} && echo {}"

# optional:  bgzip outputs
ls $out/*.bed | xargs -I {} sh -c "bgzip {} && echo {}"

# clean up
rm raw.tgz
rm *_mnemonics.bed.gz

# check
ls $out | wc -l  # 1905
```

**For hg19** compatible, avoid the liftover:

```sh
# all_hg38lift.mnemonics.bedFiles.tgz  -->  all.mnemonics.bedFiles.tgz
wget http://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/all.mnemonics.bedFiles.tgz -O raw.tgz
# for this version, there should be 55,605,005 intervals instead
```

# rme_small

This is just a subset of the `roadmap_epigenomics` used for diagnostics. This setup assumes you already have the larger dataset ready.

```sh
rme="data/roadmap_epigenomics/beds"
rme_small="data/rme_small/beds"

# 1. copy a subset of bed files
mkdir $rme_small
ls $rme | head -n 56 | xargs -I{} cp $rme/{} $rme_small

# 2. unzip if necessary
gunzip $rme_small/*.gz;

# 3. take a subset of intervals
cd $rme_small
ls | xargs -I {} sh -c "head -n 1000 {} > {}_ && mv {}_ {}"
```
