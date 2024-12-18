import intervalTransforms as IT
from dataWrangling.listDataset import ListDataset
from dataWrangling.transformDataset import *


class IdxTransform(Transform):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, item, idx):
        return idx


class VerboseTransform(Transform):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, item, idx):
        return (item, idx)


def test_transform_dataset():
    backingDs = ListDataset([1, 2, 3])
    transformedDs = TransformDataset(backingDs, [
        IdxTransform(2),
        IdxTransform(1),
        IdxTransform(3),
        IdxTransform(1),
    ])

    assert len(transformedDs) == 18

    for i in range(len(transformedDs)):
        assert transformedDs[i] == i

    backingDs = ListDataset([2, 3])
    transformedDs = TransformDataset(backingDs, [
        VerboseTransform(2)
    ])

    target = [
        (2, 0),
        (2, 1),
        (3, 2),
        (3, 3)
    ]

    for i in range(len(transformedDs)):
        assert transformedDs[i] == target[i]

    backingDs = ListDataset([2, 3])
    transformedDs = TransformDataset(backingDs, [
        VerboseTransform(2),
        VerboseTransform(2),
    ])

    target = [
        ((2, 0), 0),
        ((2, 0), 1),
        ((2, 1), 2),
        ((2, 1), 3),
        ((3, 2), 4),
        ((3, 2), 5),
        ((3, 3), 6),
        ((3, 3), 7),
    ]

    for i in range(len(transformedDs)):
        assert transformedDs[i] == target[i]


def test_interval_transforms():
    baseInt = (None, 3, 6)

    translated = IT.Translate(1)(baseInt, 0)
    assert translated[2] == 7

    swelled = IT.Swell(3)(baseInt, 0)
    assert swelled[1] == 0
    assert swelled[2] == 9

    chunk0 = IT.Chunk(3)(swelled, 0)
    chunk1 = IT.Chunk(3)(swelled, 1)
    chunk2 = IT.Chunk(3)(swelled, 2)
    assert chunk0 == (None, 0, 3)
    assert chunk1 == (None, 3, 6)
    assert chunk2 == (None, 6, 9)
