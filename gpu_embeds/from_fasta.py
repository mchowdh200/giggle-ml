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
    seq = seq.upper() # aCgT -> ACGT
    fastaContent.append(seq)

bedContent = []
with open(bedPath) as f:
    for line in f:
        name, start, stop = line.split()[:3]
        start = int(start)
        stop = int(stop)

        if name not in nameMap:
            raise ValueError(f"Chromosome name, \"{name}\", not recognized.")

        chrm = nameMap[name]
        seq = fastaContent[chrm][start-1 : stop]
        bedContent.append(seq)

max_length = len(max(bedContent))
max_length = 500
bedTokenized = []
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
    model_max_length=max_length + 2,  # to account for special tokens, like EOS
    add_special_tokens=False,  # we handle special tokens elsewhere
    padding_side='left', # since HyenaDNA is causal, we pad on the left
)

bedContentFiltered = list(filter(lambda x: x, bedContent))
if len(bedContentFiltered) != len(bedContent):
    print(str(len(bedContent) - len(bedContentFiltered)) +
        " intervals in BED file did not map to a sequence." +
        " Discarding bad entries.\n" + 
        str(len(bedContentFiltered)) +
        " entries mapped successfully. ")

for seq in bedContentFiltered:
    tok = tokenizer(seq,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True
    )

    tok = tok['input_ids']
    tok = torch.LongTensor(tok).unsqueeze(0)  # unsqueeze for batch dim
    bedTokenized.append(tok)

class BedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(bedTokenized)

    def __getitem__(self, idx):
        return bedTokenized[idx]

infer(BedDataset())
