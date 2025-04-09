from segma.models.base import ConvolutionSettings


def test_rf_start_i() -> None:
    cs = ConvolutionSettings((3, 2), (3, 1), (1, 0))
    cs_2 = ConvolutionSettings((2,), (1,), (0,))

    assert cs.rf_start_i(0) == -1
    # Test for the receptive field at the 2nd layer
    assert cs_2.rf_start_i(0) == 0


def test_rf_end_i() -> None:
    cs = ConvolutionSettings((3, 2), (3, 1), (1, 0))
    cs_2 = ConvolutionSettings((2,), (1,), (0,))

    assert cs.rf_end_i(0) == 4
    # Test for the receptive field at the 2nd layer
    assert cs_2.rf_end_i(0) == 1


def test_rf_size() -> None:
    cs = ConvolutionSettings((3, 2), (3, 1), (1, 0))
    cs_2 = ConvolutionSettings((2,), (1,), (0,))

    assert cs.rf_size() == 6
    # Test for the receptive field at the 2nd layer
    assert cs_2.rf_size() == 2
