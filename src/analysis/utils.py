from collections.abc import Sequence

import numpy as np


def confInt95(item: Sequence[float] | np.ndarray):
    if not isinstance(item, np.ndarray):
        item = np.array(item)
    return (1.96 * np.std(item) / np.sqrt(len(item))).item()
