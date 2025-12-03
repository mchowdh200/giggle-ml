from pathlib import Path

import giggleml.utils.roadmap_epigenomics as rme
from giggleml.data_wrangling.interval_dataset import BedDataset
from giggleml.utils.print_utils import progress_logger


def main():
    with progress_logger(len(rme.bed_names), "reading") as ckpt:
        with open("tmp", "w") as f:
            lines = []

            for x in rme.bed_names:
                path = Path("data", "roadmap_epigenomics", "beds", x + ".bed.gz")
                lines.append(str(len(BedDataset(path))))
                ckpt()

            f.writelines(", \n".join(lines))


if __name__ == "__main__":
    main()
