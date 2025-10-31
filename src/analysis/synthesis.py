import random
from random import choice, randint
from time import time

chrm_sizes_f = [
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

chrm_sizes: list[int] = list(map(lambda x: round(x * 1e6), chrm_sizes_f))  # In Mbp
chrm_names = [f"chr{i + 1}" for i, _ in enumerate(chrm_sizes)]
chrm_names[22] = "chrX"
chrm_names[23] = "chrY"


class Chromosome:
    def __init__(self, block_size):
        self.blocks = dict()
        self.block_size = block_size

    def block_id(self, pos):
        return pos // self.block_size

    def blocks_in(self, start, end):
        start_block = self.block_id(start)
        end_block = self.block_id(end - 1)
        return range(start_block, end_block)

    def fill_block(self, block_id):
        if block_id not in self.blocks:
            entry = [0] * self.block_size
            entry = list(map(lambda x: randint(0, 3), entry))
            self.blocks[block_id] = entry

    def include(self, start, end):
        for block_id in self.blocks_in(start, end):
            self.fill_block(block_id)

    def block_as_seq(self, block_id):
        literals = "ACGT"
        block = self.blocks[block_id]
        return list(map(lambda x: literals[x], block))

    def fasta_format(self, chrm_name):
        out = [">", str(chrm_name), "\n"]

        prev_block_id = 0
        for block_id in sorted(self.blocks.keys()):
            gap = max(0, block_id - prev_block_id - 1)
            out += ["N"] * (gap * self.block_size)
            out += self.block_as_seq(block_id)
            prev_block_id = block_id

        return "".join(out)

    def __repr__(self):
        return str(self.blocks)


def synthesize(fasta_out, out_files, seq_len_min, seq_len_max, seq_per_universe, seed):
    random.seed(seed)
    print("Using chromosome structure:", list(zip(chrm_names, chrm_sizes)))

    # ---------------- Synthesize intervals ----------------
    print("Synthesizing intervals...")
    universes = []
    for _ in range(len(out_files)):
        univ = []

        for _ in range(seq_per_universe):
            seq_len = randint(seq_len_min, seq_len_max)
            chrom_id = randint(0, len(chrm_sizes) - 1)
            chrm_size = chrm_sizes[chrom_id]
            start = randint(0, chrm_size - seq_len)
            end = start + seq_len
            univ.append((chrom_id, start, end))
        universes.append(univ)

    # -------------- Synthesize nucleotides ----------------
    print("Synthesizing nucleotides...")
    block_size = seq_len_max
    blocks = [Chromosome(block_size) for _ in range(len(chrm_sizes))]

    t0 = time()
    for univ_id, univ in enumerate(universes):
        print("Building universe", univ_id)

        for reg_id, (chrm_id, start, end) in enumerate(univ):
            # extra padding
            size = end - start
            start = max(0, start - size)
            end = min(chrm_sizes[chrm_id], end + size)

            if reg_id % 1000 == 0:
                dt = time() - t0
                eta = (dt / (reg_id + 1)) * (seq_per_universe - reg_id)
                eta = round(eta / 60, 1)
                print(f"- region {reg_id},\tETA: {eta} min")

            chrm = blocks[chrm_id]
            chrm.include(start, end)

    # ---------------- File output -----------------------
    # TODO: check paths exist before attempting long computation
    print("Generating output files...")
    # bed files
    for reg_id, reg in enumerate(universes):
        path = out_files[reg_id]
        with open(path, "w") as f:
            for chrm_id, start, end in reg:
                chrm_name = chrm_names[chrm_id]
                f.write(f"{chrm_name}\t{start}\t{end}\n")

    # fasta file
    with open(fasta_out, "w") as f:
        out = []
        for chrm_id, chrm in enumerate(blocks):
            chrm_name = chrm_names[chrm_id]
            out.append(chrm.fasta_format(chrm_name))
            out.append("\n")
        f.write("".join(out))


def all_random_fasta(fasta_out, seed: int = 42):
    random.seed(seed)
    print("Using chromosome structure:", list(zip(chrm_names, chrm_sizes)))

    with open(fasta_out, "w") as file:
        for name, size in zip(chrm_names, chrm_sizes):
            size += 750e3  # spare, to accommodate rounding
            _nucleotide = lambda: choice("ACGT")
            content = [_nucleotide() for _ in range(int(size))]
            file.write(f">{name}\n")
            file.write("".join(content) + "\n")
            print(f"finished {name}")


def main():
    all_random_fasta("data/hg/synthetic.fa")


if __name__ == "__main__":
    main()
