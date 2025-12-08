Seqpare ranks are used in benchmarking and during training, usually in conjunction with roadmap epigenomics bed files.

# roadmap_epigenomics/seqpare_ranks

Seqpare self-similarity

For a directory of .bed files, `beds/`  
And an output directory, `seqpare_ranks/`

**/!\\ Make sure `seqpare` is installed. https://github.com/deepstanding/seqpare**

```sh
parallel seqpare "'beds/*.bed'" {} -m 1 -o seqpare_ranks/{/.}.tsv ::: beds/*.bed
```

This may take several minutes
