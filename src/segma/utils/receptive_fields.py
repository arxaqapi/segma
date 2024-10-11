from math import prod


def rf_start_i(u_L: int, strides: list[int], paddings: list[int], L: int) -> int:
    """Computes the start index of the receptive field.

    see eq (5) in https://distill.pub/2019/computing-receptive-fields/

    Args:
        u_L (int): start index of the output range.
        strides (list[int]): list of stride sizes, from first to last convolutionnal layer.
        paddings (list[int]): list of padding values (size of the right and left padding), from first to last convolutionnal layer.
        L (int): number of layers to take into account.

    Returns:
        int: Start index of the receptive field in the input vector. Can be negative.
    """
    S_0 = prod(strides)

    P_0 = 0
    for layer_i in range(L):
        P_0 += paddings[layer_i] * prod(strides[:layer_i])

    return u_L * S_0 - P_0


def rf_end_i(
    v_L: int, kernels: list[int], strides: list[int], paddings: list[int], L: int
) -> int:
    """Computes the end index of the receptive field.

    see eq (6) in https://distill.pub/2019/computing-receptive-fields/

    Args:
        v_L (int): end index of the output range.
        kernels (list[int]): list of kernel sizes, from first to last convolutionnal layer.
        strides (list[int]): list of stride sizes, from first to last convolutionnal layer.
        paddings (list[int]): list of padding values (size of the right and left padding), from first to last convolutionnal layer.
        L (int): number of layers to take into account.

    Returns:
        int: End index of the receptive field in the input vector. Can be greater than the size of the input vector.
    """
    S_0 = prod(strides)

    rt = 0
    for layer_i in range(L):
        rt += (1 + paddings[layer_i] - kernels[layer_i]) * prod(strides[:layer_i])

    return v_L * S_0 - rt


def rf_size(kernels: list[int], strides: list[int], L: int) -> int:
    """Computes the size of the receptive field.

    see eq (2) in https://distill.pub/2019/computing-receptive-fields/

    Args:
        kernels (list[int]): list of kernel sizes, from first to last convolutionnal layer.
        strides (list[int]): list of stride sizes, from first to last convolutionnal layer.
        L (int): number of layers to take into account.

    Returns:
        int: Size of the receptive field.
    """

    rf = 0
    for layer_i in range(L):
        rf += (kernels[layer_i] - 1) * prod(strides[:layer_i])
    return rf + 1
