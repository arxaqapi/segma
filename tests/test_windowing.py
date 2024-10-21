import numpy as np
from interlap import InterLap

from segma.dataloader import windows_to_targets
from segma.utils.encoders import PowersetMultiLabelEncoder


def test_windows_to_target():
    l_enc = PowersetMultiLabelEncoder(("child", "female", "male"))

    # generate windows
    wins = np.array(
        [
            [0, 10],
            [5, 15],
            [10, 20],
            [15, 25],
            [20, 30],
            [25, 35],
            [30, 40],
            [35, 45],
            [40, 50],
        ]
    )
    # hardcode labels
    labels = InterLap([(0, 10, "child"), (5, 15, "male"), (40, 50, "female")])

    targets = windows_to_targets(windows=wins, label_encoder=l_enc, labels=labels)

    assert int(targets[4][0]) == 1
    assert int(targets[5][0]) == 1

    # test that for frames with no overlap (no class), the default class is returned
