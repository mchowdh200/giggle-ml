from dataWrangling import fasta
from dataWrangling.intervalDataset import BedDataset
from dataWrangling.slidingWindowDataset import SlidingWindowDataset


def testBedFastaParsing():
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    assert bed[0] == ("chr1", 0, 1)
    assert bed[1] == ("chr1", 0, 40)
    assert bed[4] == ("chr1", 15, 25)
    assert bed[5] == ("chr1", 10, 30)
    assert bed[6] == ("chr2", 10, 30)
    assert bed[-1] == ("chr3", 5, 10)
    assert bed[-2] == ("chr3", 0, 5)

    fa = fasta.map(bed)
    assert len(fa) == len(bed)
    assert fa[-1] == "TTTTT"
    assert fa[-2] == "AAAAA"


def testSlidingWindowDataset():
    # TODO: SlidingWindowDataset requires further tests.
    # there's weird stuff going on

    content = [
        "Steaks are best hot!",
        "123456789",
    ]
    dataset = SlidingWindowDataset(content, 0.5)
    expecting = [
        "Steaks are",
        "s are best",
        " best hot!",
        "1234",
        "3456",
        "5678",
    ]

    for i, expect in enumerate(expecting):
        assert expect == dataset[i]
