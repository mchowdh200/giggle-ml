from gpu_embeds.inference_batch import batchInfer
from gpu_embeds.standalone_hyenadna import CharacterTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from collections import deque
from functools import cached_property
from pyfaidx import Fasta


class BedDataset(torch.utils.data.Dataset):
    def __init__(self, bedPath, inMemory=True, bufferSize=100, limit=None):
        self.bedPath = bedPath
        self.inMemory = inMemory
        self.bufferSize = bufferSize
        limit = float('inf') if limit is None else limit

        if inMemory:
            self.bedContent = []
            with open(bedPath) as f:
                for i, line in enumerate(f):
                    if i < limit:
                        name, start, stop = line.split()[:3]
                        start = int(start)
                        stop = int(stop)
                        self.bedContent.append((name, start, stop))
                        continue
                    break
            self.length = len(self.bedContent)
        else:
            self.length = min(sum(1 for line in open(bedPath)), limit)
            self.bedBuffer = {}
            self.queue = deque()

    def __len__(self):
        return self.length

    def fetch(self, idx):
        for i, line in enumerate(open(self.bedPath)):
            if i >= idx + self.bufferSize // 2:
                break
            if i >= idx:
                if i not in self.bedBuffer:
                    name, start, stop = line.split()[:3]
                    start = int(start)
                    stop = int(stop)
                    self.bedBuffer[i] = (name, start, stop)
                    self.queue.append(i)

        while len(self.queue) > self.bufferSize:
            self.bedBuffer.pop(self.queue.popleft())

    def __getitem__(self, idx):
        if self.inMemory:
            return self.bedContent[idx]
        else:
            if idx not in self.bedBuffer:
                self.fetch(idx)
            return self.bedBuffer[idx]


class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, fastaPath, bedDataset):
        self.fastaPath = fastaPath
        self.bedDataset = bedDataset

    @cached_property
    def index(self):
        # why one_based_attributes=True by default?
        return Fasta(self.fastaPath, one_based_attributes=False)

    def __len__(self):
        return len(self.bedDataset)

    def __getitem__(self, idx):
        name, start, stop = self.bedDataset[idx]
        if name not in self.index:
            raise ValueError(f"Chromosome name, \"{name}\", not recognized.")

        seq = self.index[name][start:stop].seq

        if len(seq) == 0:
            raise ValueError(f"Failed fasta mapping for {name}:{start}-{stop}")

        return seq


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, fastaDataset):
        self.fastaDataset = fastaDataset
        self.padToLength = 500

    @cached_property
    def tokenizer(self):
        return CharacterTokenizer(
            # add DNA characters, N is uncertain
            characters=['A', 'C', 'G', 'T', 'N'],
            # to account for special tokens, like EOS
            model_max_length=self.padToLength + 2,
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left',  # since HyenaDNA is causal, we pad on the left
        )

    def __len__(self):
        return len(self.fastaDataset)

    def __getitem__(self, idx):
        seq = self.fastaDataset[idx]

        # Begin tokenization
        seq = seq.upper()  # aCgT -> ACGT
        tok = self.tokenizer(seq,
                             add_special_tokens=False,
                             padding="max_length",
                             max_length=self.padToLength,
                             truncation=True)

        tok = tok['input_ids']
        return torch.LongTensor(tok)


# TODO:! Inference is much slower when multiple workers, synchronization issue?
