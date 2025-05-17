from collections.abc import Sequence

import numpy as np


def confInt95(item: Sequence[float]):
    return (1.96 * np.std(item) / np.sqrt(len(item))).item()
