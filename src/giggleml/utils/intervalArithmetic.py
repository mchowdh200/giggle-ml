from giggleml.utils.types import GenomicInterval


def intersect(x: GenomicInterval, y: GenomicInterval) -> GenomicInterval | None:
    ch1, start1, end1 = x
    ch2, start2, end2 = y

    if ch1 != ch2:
        return None

    start = max(start1, start2)
    end = min(end1, end2)

    if start >= end:
        return None

    return (ch1, start, end)
