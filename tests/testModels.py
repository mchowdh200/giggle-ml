import torch

from giggleml.embedGen.embedModel import HyenaDNA
from giggleml.utils.guessDevice import guessDevice


def testHyenaDNA():
    model = HyenaDNA("1k")
    model.to(guessDevice())
    results = model.embed(["ACGTTGCA", "ACT"]).cpu()

    assert len(results) == 2

    for embed in results:
        assert len(embed) == model.embedDim
        assert embed.dtype == torch.float32

    brief = results[:, ::10]
    brief = torch.round(brief * 10)

    # INFO: IF using mean aggregation
    expect = torch.Tensor(
        [
            [2.0, 6.0, -16.0, -1.0, -9.0, -5.0, 10.0, 7.0, -1.0, 1.0, -3.0, 9.0, -11.0],
            [2.0, 6.0, -16.0, -1.0, -9.0, -5.0, 10.0, 7.0, -1.0, 1.0, -3.0, 9.0, -11.0],
        ]
    )

    # INFO: IF using mean pooling (aggregation)
    # expect = torch.Tensor(
    #     [
    #         [-2.0, 6.0, -7.0, 6.0, 1.0, -14.0, -2.0, 5.0, 3.0, 5.0, 0.0, 12.0, -19.0],
    #         [-3.0, 6.0, -7.0, 7.0, 1.0, -13.0, -2.0, 5.0, 2.0, 6.0, 0.0, 13.0, -19.0],
    #     ]
    # )

    assert torch.equal(brief, expect)


# INFO: CountACGT Model already tested implicitly in testDataWrangling.py


def testRegion2Vec():
    # TODO: impl R2V test
    ...
