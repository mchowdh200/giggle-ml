from statistical_tests import tests
from gpu_embeds.from_fasta import generate_embeddings


# TODO: repair RCT & CTT


def get_intervals(bedPath):
    with open(bedPath) as f:
        intervals = []
        for line in f:
            columns = line.strip().split()
            chromosome, start, end, *_ = columns
            start = int(start)
            end = int(end)
            intervals.append((chromosome, start, end))
    return intervals


# from gpu_embeds.from_genomic_benchmarks import main
def main():

    # def main_():
    # TODO: hoist defaults into generate_embeddings fn signature?
    limit = 100
    batchSize = 5
    # fastaPath = "./data/synthetic/seqs.fa"
    # bedPath = "./data/synthetic/universe_0.bed"
    fastaPath = "./data/hg38.fa"
    bedPath = "./data/hg38_trf.bed"
    # outPath = "./data/synthetic/uni_0_embeds.npy"
    outPath = None

    intervals = get_intervals(bedPath)[:limit]
    embeds = generate_embeddings(
        fastaPath,
        bedPath,
        batchSize=batchSize,
        limit=limit)

    print()
    tests.gdst(intervals, embeds)
    tests.npt(intervals, embeds)
    tests.cct(intervals, embeds)


if __name__ == "__main__":
    main()
