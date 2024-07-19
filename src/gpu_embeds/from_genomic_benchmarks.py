import torch
import numpy as np
from gpu_embeds.inference_batch import batchInfer
from gpu_embeds.truncated_dataset import TruncatedDataset
from gpu_embeds.standalone_hyenadna import CharacterTokenizer
from gpu_embeds.genomic_benchmark_dataset import GenomicBenchmarkDataset


# TODO: extract (from_[fasta, genomicbench]) similarity
def prepareDataset(limit):
    # let's fix the max_length (to reduce the padding amount, conserves memory)
    max_length = 500

    # create tokenizer
    tokenizer = CharacterTokenizer(
        # add DNA characters, N is uncertain
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left',  # since HyenaDNA is causal, we pad on the left
    )

    # Sample small dataloader w/ GenomicBenchmarks

    # data settings:
    # we need to choose the dataset and batch size to loop thru
    dataset_name = 'human_enhancers_cohn'
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # TODO: The tokenizer should be seperated from this dataset object
    dataset = GenomicBenchmarkDataset(
        max_length=max_length,
        use_padding=use_padding,
        split='test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=rc_aug,
        add_eos=add_eos)

    return TruncatedDataset(dataset, limit)


def main():
    # INFO: Config
    # outFile = "./data/embeddings.npy"
    outFile = None
    batchSize = 10
    limit = 100

    results = batchInfer(prepareDataset(limit), batchSize)
    print("Success.")

    if outFile:
        print("Serializing embeddings...")
        np.save(results, outFile)
