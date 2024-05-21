from inference_batch import infer
from torch.utils.data import DataLoader
from Bio import SeqIO
import torch
from standalone_hyenadna import CharacterTokenizer


fastaPath = "./hg38.fa"
bedPath = "./hg38.bed"

#  chrm id --> SeqIO.SeqRecord object
fastaContent = list()
nameMap = dict()

for entry in SeqIO.parse(open(fastaPath), 'fasta'):
    name = entry.name
    seq = None

    try:
        # The old way, removed in Biopython 1.73
        seq = entry.tostring()
    except AttributeError:
        # The new way, needs Biopython 1.45 or later.
        # Don't use this on Biopython 1.44 or older as truncates
        seq = str(entry)

    nameMap[name] = len(fastaContent)
    fastaContent.append(seq)

chrmLen = len(max(fastaContent))
fastaTokenized = [None]*len(fastaContent)
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
    model_max_length=chrmLen + 2,  # to account for special tokens, like EOS
    add_special_tokens=False,  # we handle special tokens elsewhere
    padding_side='left', # since HyenaDNA is causal, we pad on the left
)

for i, seq in enumerate(fastaContent):
    # defaults from hyenadna
    seq = tokenizer(seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=chrmLen,
        truncation=True,
    )

    seq = seq["input_ids"]  # get input_ids
    seq = torch.LongTensor(seq)
    fastaTokenized[i] = seq

fastaTensor = torch.stack(fastaTokenized)
fastaTensor.to(device)

bedContent = []
with open(bedPath) as f:
    for line in f:
        entry = line.split()[:3]
        name = entry[0]

        if name not in nameMap:
            raise f"Chromosome name, \"{name}\", not recognized."

        tensor = torch.tensor(entry, dtype=torch.int)
        bedContent.append(entry)

bedTensor = torch.tensor(bedContent)

class BedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(bedTensor)

    def __getitem__(self, idx):
        return sbedTensor[idx]

