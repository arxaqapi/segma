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

    assert cs.rf_size == 6
    # Test for the receptive field at the 2nd layer
    assert cs_2.rf_size == 2


def test_n_windows() -> None:
    cs = ConvolutionSettings((320,), (320,), (0,))

    assert cs.n_windows(4 * 16_000, strict=False) == 200
    assert cs.n_windows(4 * 16_000, strict=True) == 200


def test_n_windows_hubert() -> None:
    cs_hubert = ConvolutionSettings(
        kernels=(10, 3, 3, 3, 3, 2, 2),
        strides=(5, 2, 2, 2, 2, 2, 2),
        paddings=(0, 0, 0, 0, 0, 0, 0),
    )

    assert cs_hubert.n_windows(4 * 16_000, strict=False) == 200
    assert cs_hubert.n_windows(4 * 16_000, strict=True) == 199


def test_n_windows_hubert_simplified() -> None:
    cs = ConvolutionSettings((400,), (320,), (0,))

    assert cs.n_windows(4 * 16_000, strict=False) == 200
    assert cs.n_windows(4 * 16_000, strict=True) == 199


def test_n_windows_hubert_context() -> None:
    cs = ConvolutionSettings((400,), (320,), (0,))

    for context in range(2, 31, 2):
        cd = context * 16_000 / (0.02 * 16_000)
        assert cs.n_windows(context * 16_000, strict=False) == cd
        assert cs.n_windows(context * 16_000, strict=True) == cd - 1
