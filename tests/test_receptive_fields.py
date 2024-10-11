from segma.utils.receptive_fields import rf_end_i, rf_size, rf_start_i


def test_rf_start_i() -> None:
    _kernels = [3, 2]
    paddings = [1, 0]
    strides = [3, 1]

    assert rf_start_i(0, strides=strides, paddings=paddings, L=2) == -1
    # Test for the receptive field at the 2nd layer
    assert rf_start_i(0, strides=strides[1:], paddings=paddings[1:], L=1) == 0


def test_rf_end_i() -> None:
    kernels = [3, 2]
    paddings = [1, 0]
    strides = [3, 1]

    assert rf_end_i(0, kernels=kernels, strides=strides, paddings=paddings, L=2) == 4
    # Test for the receptive field at the 2nd layer
    assert (
        rf_end_i(
            0, kernels=kernels[1:], strides=strides[1:], paddings=paddings[1:], L=1
        )
        == 1
    )


def test_rf_size() -> None:
    kernels = [3, 2]
    paddings = [1, 0]
    strides = [3, 1]

    assert rf_size(kernels=kernels, strides=strides, L=2) == 6
    # Test for the receptive field at the 2nd layer
    assert rf_size(kernels=kernels[1:], strides=strides[1:], L=1) == 2
