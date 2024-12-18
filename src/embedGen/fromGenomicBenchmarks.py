import numpy as np

from dataWrangling.genomicBenchmarkDataset import GenomicBenchmarkDataset
from dataWrangling.truncatedDataset import TruncatedDataset
from embedGen.inferenceBatch import BatchInferHyenaDNA
from embedGen.standaloneHyenadna import CharacterTokenizer


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


def main(limit=100, batchSize=10, outFile=None):
    inf = BatchInferHyenaDNA()
    results = inf.batchInfer(prepareDataset(limit), outFile, batchSize)
    print("Success.")

    if outFile:
        print("Serializing embeddings...")
        np.save(outFile, results)
    return results
