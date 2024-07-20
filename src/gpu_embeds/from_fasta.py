from gpu_embeds.inference_batch import batchInfer
from gpu_embeds.standalone_hyenadna import CharacterTokenizer
from torch.utils.data import DataLoader
from Bio import SeqIO
import torch
import sys
import numpy as np


# TODO: extract
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, contents):
        self.contents = contents

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        return self.contents[idx]


def generate_embeddings(fastaPath, bedPath, batchSize=16, outPath=None, limit=None):
    #  chrm id --> SeqIO.SeqRecord object
    fastaContent = list()
    nameMap = dict()

    print("Parsing inputs...")
    if limit is not None:
        print(f"Limiting to {limit} sequences.")

    for entry in SeqIO.parse(open(fastaPath), 'fasta'):
        name = entry.id
        seq = entry.seq

        try:
            # The old way, removed in Biopython 1.73
            seq = seq.tostring()
        except AttributeError:
            # The new way, needs Biopython 1.45 or later.
            # Don't use this on Biopython 1.44 or older as truncates
            seq = str(seq)

        nameMap[name] = len(fastaContent)
        seq = seq.upper()  # aCgT -> ACGT
        fastaContent.append(seq)

    bedContent = []
    with open(bedPath) as f:
        for line in f:
            if limit is not None and limit == 0:
                break
            limit -= 1

            name, start, stop = line.split()[:3]
            start = int(start)
            stop = int(stop)

            if name not in nameMap:
                raise ValueError(
                    f"Chromosome name, \"{name}\", not recognized.")

            chrm = nameMap[name]
            seq = fastaContent[chrm][start-1: stop]
            bedContent.append(seq)

    max_length = len(max(bedContent))
    max_length = 500
    bedTokenized = []
    tokenizer = CharacterTokenizer(
        # add DNA characters, N is uncertain
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left',  # since HyenaDNA is causal, we pad on the left
    )

    bedContentFiltered = list(filter(lambda x: x, bedContent))
    if len(bedContentFiltered) != len(bedContent):
        print(str(len(bedContent) - len(bedContentFiltered)) +
              " intervals in BED file did not map to a sequence." +
              " Discarding bad entries.\n" +
              str(len(bedContentFiltered)) +
              " entries mapped successfully. ")

    print('Tokenizing...')
    for seq in bedContentFiltered:
        tok = tokenizer(seq,
                        add_special_tokens=False,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True
                        )

        # TODO: print dict keys, is there a mask? --> embedding aggregation
        tok = tok['input_ids']
        tok = torch.LongTensor(tok)
        # tok = torch.LongTensor(tok).unsqueeze(0)  # unsqueeze for batch dim
        bedTokenized.append(tok)

    dataset = ListDataset(bedTokenized)
    results = batchInfer(dataset, batchSize)
    print("Success.")

    if outPath:
        print("Serializing embeddings...")
        np.save(outPath, results)
    return results
