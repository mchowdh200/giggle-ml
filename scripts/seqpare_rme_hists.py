import os
import re
from pathlib import Path

from matplotlib import pyplot as plt


def parse(line: str):
    items = line.strip().split("\t")
    assert len(items) >= 6
    raw_label = items[5]
    assert (match := re.match(r"(.+/)*(.+)\.bed(\.gz)?", raw_label))
    label = match.group(2)
    similarity = float(items[4])
    return label, similarity


def main():
    base = Path("data", "roadmap_epigenomics", "seqpareRanks")

    for name in os.listdir(str(base)):
        with open(Path(base, name), "r") as lines:
            next(lines)  # skip header
            items = [parse(line) for line in lines]

        items.sort(key=lambda x: x[1])
        [print(x) for x in items]
        similarities = [x[1] for x in items]

        # percent smaller than .05
        whats_small = 0.05
        small = sum([x[1] < whats_small for x in items]) / 1905 * 100
        # max item
        top = max([x[1] for x in items])

        plt.hist(similarities, bins=20)
        plt.title(f"{name} histogram")
        plt.xlabel("similarity")
        plt.ylabel("frequency")
        plt.suptitle(f"{small:.1f}% < {whats_small}, max {top:.3f}")
        plt.show()

        # 90% percentile
        percentile_10 = round(similarities[round(len(similarities) * 0.1)], 2)
        percentile_90 = round(similarities[round(len(similarities) * 0.9)], 2)
        plt.plot(similarities)
        plt.title(f"{name} sorted similarities")
        plt.xlabel("(sorted) index")
        plt.ylabel("similarity")
        plt.suptitle(
            f"{percentile_10}-{percentile_90} 10%-90% percentiles in similarity"
        )
        plt.show()


if __name__ == "__main__":
    main()
