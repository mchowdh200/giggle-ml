import random
from random import randint
from time import time

chrmSizesF = [
    248.96,  # Chromosome 1
    242.19,  # Chromosome 2
    198.30,  # Chromosome 3
    190.21,  # Chromosome 4
    181.54,  # Chromosome 5
    170.81,  # Chromosome 6
    159.35,  # Chromosome 7
    145.14,  # Chromosome 8
    138.39,  # Chromosome 9
    133.80,  # Chromosome 10
    135.09,  # Chromosome 11
    133.28,  # Chromosome 12
    114.36,  # Chromosome 13
    107.04,  # Chromosome 14
    101.99,  # Chromosome 15
    90.34,  # Chromosome 16
    83.26,  # Chromosome 17
    80.37,  # Chromosome 18
    58.62,  # Chromosome 19
    64.44,  # Chromosome 20
    46.71,  # Chromosome 21
    50.82,  # Chromosome 22
    156.04,  # Chromosome X
    57.23,  # Chromosome Y
]

chrmSizes: list[int] = list(map(lambda x: round(x * 1e6), chrmSizesF))  # In Mbp
chrmNames = [f"chr{i+1}" for i, _ in enumerate(chrmSizes)]
chrmNames[22] = "chrX"
chrmNames[23] = "chrY"


class Chromosome:
    def __init__(self, blockSize):
        self.blocks = dict()
        self.blockSize = blockSize

    def block_id(self, pos):
        return pos // self.blockSize

    def blocks_in(self, start, end):
        startBlock = self.block_id(start)
        endBlock = self.block_id(end - 1)
        return range(startBlock, endBlock)

    def fill_block(self, blockId):
        if blockId not in self.blocks:
            entry = [0] * self.blockSize
            entry = list(map(lambda x: randint(0, 3), entry))
            self.blocks[blockId] = entry

    def include(self, start, end):
        for blockId in self.blocks_in(start, end):
            self.fill_block(blockId)

    def block_as_seq(self, blockId):
        literals = "ACGT"
        block = self.blocks[blockId]
        return list(map(lambda x: literals[x], block))

    def fasta_format(self, chrmName):
        out = [">", str(chrmName), "\n"]

        prevBlockId = 0
        for blockId in sorted(self.blocks.keys()):
            gap = max(0, blockId - prevBlockId - 1)
            out += ["N"] * (gap * self.blockSize)
            out += self.block_as_seq(blockId)
            prevBlockId = blockId

        return "".join(out)

    def __repr__(self):
        return str(self.blocks)


def synthesize(fastaOut, outFiles, seqLenMin, seqLenMax, seqPerUniverse, seed):
    random.seed(seed)
    print("Using chromosome structure:", list(zip(chrmNames, chrmSizes)))

    # ---------------- Synthesize intervals ----------------
    print("Synthesizing intervals...")
    universes = []
    for _ in range(len(outFiles)):
        univ = []

        for _ in range(seqPerUniverse):
            seqLen = randint(seqLenMin, seqLenMax)
            chromId = randint(0, len(chrmSizes) - 1)
            chrmSize = chrmSizes[chromId]
            start = randint(0, chrmSize - seqLen)
            end = start + seqLen
            univ.append((chromId, start, end))
        universes.append(univ)

    # -------------- Synthesize nucleotides ----------------
    print("Synthesizing nucleotides...")
    blockSize = seqLenMax
    blocks = [Chromosome(blockSize) for _ in range(len(chrmSizes))]

    t0 = time()
    for univId, univ in enumerate(universes):
        print("Building universe", univId)

        for regId, (chrmId, start, end) in enumerate(univ):
            # extra padding
            size = end - start
            start = max(0, start - size)
            end = min(chrmSizes[chrmId], end + size)

            if regId % 1000 == 0:
                dt = time() - t0
                eta = (dt / (regId + 1)) * (seqPerUniverse - regId)
                eta = round(eta / 60, 1)
                print(f"- region {regId},\tETA: {eta} min")

            chrm = blocks[chrmId]
            chrm.include(start, end)

    # ---------------- File output -----------------------
    # TODO: check paths exist before attempting long computation
    print("Generating output files...")
    # bed files
    for regId, reg in enumerate(universes):
        path = outFiles[regId]
        with open(path, "w") as f:
            for chrmId, start, end in reg:
                chrmName = chrmNames[chrmId]
                f.write(f"{chrmName}\t{start}\t{end}\n")

    # fasta file
    with open(fastaOut, "w") as f:
        out = []
        for chrmId, chrm in enumerate(blocks):
            chrmName = chrmNames[chrmId]
            out.append(chrm.fasta_format(chrmName))
            out.append("\n")
        f.write("".join(out))
