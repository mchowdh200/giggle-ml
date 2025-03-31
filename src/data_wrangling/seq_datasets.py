from collections import deque

import torch
from pyparsing.core import cached_property
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BedDataset(torch.utils.data.Dataset):
    def __init__(self, bedPath, inMemory=True, bufferSize=100, rowLimit=None, maxLen=None):
        self.bedPath = bedPath
        self.inMemory = inMemory
        self.bufferSize = bufferSize

        rowLimit = float('inf') if rowLimit is None else rowLimit
        self.maxLen = float('inf') if maxLen is None else maxLen

        if inMemory:
            self._getSeq = self._fetchMemory
        else:
            self.length = min(sum(1 for line in open(bedPath)), rowLimit)
            self.bedBuffer = {}
            self.queue = deque()
            self._getSeq = self._fetchBuffer

    @cached_property
    def bedContent(self):
        content = []

        with open(self.bedPath) as f:
            for i, line in enumerate(f):
                if i < self.rowLimit:
                    name, start, stop = line.split()[:3]
                    start = int(start)
                    stop = int(stop)
                    content.append((name, start, stop))
                    continue
                break
        self.length = len(content)

    def __len__(self):
        return self.length

    def _fetchMemory(self, idx):
        return self.bedContent[idx]

    def _fetchBuffer(self, idx):
        """
        Fetch a sequence from the bed file unless it is already cached.
        :param idx:
        :return:
        """

        if idx in self.bedBuffer:
            return self.bedBuffer[idx]

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
        interval = self._getSeq(idx)
        chrom, start, stop = interval
        size = min(self.maxLen, stop - start)
        return (chrom, start, start + size)


class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, bedDataset):
        self.bedDataset = bedDataset
        self.seqs = seqs

    def __len__(self):
        return len(self.bedDataset)

    def __getitem__(self, idx):
        name, start, stop = self.bedDataset[idx]
        return self.seqs[name][start:stop]


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, fastaDataset, padToLength=500):
        self.fastaDataset = fastaDataset
        self.padToLength = padToLength
        self.modelName = 'LongSafari/hyenadna-tiny-1k-seqlen-hf' # TODO: parameterize
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName,
                                             trust_remote_code=True)

    def __len__(self):
        return len(self.fastaDataset)

    def __getitem__(self, idx):
        return self.fastaDataset[idx]

    def collate_fn(self, batch):
        batch = [ seq.upper() for seq in batch ]
        tok = self.tokenizer.batch_encode_plus(batch,
                             add_special_tokens=False,
                             padding="max_length",
                             max_length=self.padToLength,
                             truncation=True,
                             return_tensors="pt")
        return tok['input_ids']
