from dataclasses import dataclass
from functools import cached_property
from math import ceil, prod


@dataclass
class ConvolutionSettings:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]

    def __post_init__(self):
        if not (len(self.kernels) == len(self.strides) == len(self.paddings)):
            raise ValueError(
                "Given settings do not match, please provide matching dimensions for kernels, strides and paddings."
            )

    def rf_start_i(self, u_L: int) -> int:
        """Computes the start index of the receptive field.

        see eq (5) in https://distill.pub/2019/computing-receptive-fields/

        Args:
            u_L (int): start index of the output range.

        Returns:
            int: Start index of the receptive field in the input vector. Can be negative.
        """
        L = len(self.strides)
        assert L == len(self.paddings)
        S_0 = prod(self.strides)

        P_0 = 0
        for layer_i in range(L):
            P_0 += self.paddings[layer_i] * prod(self.strides[:layer_i])

        return u_L * S_0 - P_0

    def rf_end_i(self, v_L: int) -> int:
        """Computes the end index of the receptive field.

        see eq (6) in https://distill.pub/2019/computing-receptive-fields/

        Args:
            v_L (int): end index of the output range.

        Returns:
            int: End index of the receptive field in the input vector. Can be greater than the size of the input vector.
        """
        L = len(self.kernels)
        assert L == len(self.strides) == len(self.paddings)

        S_0 = prod(self.strides)

        rt = 0
        for layer_i in range(L):
            rt += (1 + self.paddings[layer_i] - self.kernels[layer_i]) * prod(
                self.strides[:layer_i]
            )

        return v_L * S_0 - rt

    @cached_property
    def rf_size(self) -> int:
        """Computes the size of the receptive field.

        see eq (2) in https://distill.pub/2019/computing-receptive-fields/

        Returns:
            int: Size of the receptive field.
        """
        L = len(self.kernels)
        assert L == len(self.strides)

        rf = 0
        for layer_i in range(L):
            rf += (self.kernels[layer_i] - 1) * prod(self.strides[:layer_i])
        return rf + 1

    def rf_center_i(self, u_L: int):
        """Center of receptive field"""
        L = len(self.kernels)
        assert L == len(self.strides) == len(self.paddings)

        S_0 = prod(self.strides)
        P_0 = 0
        for layer_i in range(L):
            P_0 += self.paddings[layer_i] * prod(self.strides[:layer_i])

        return u_L * S_0 + (self.rf_size - 1) / 2 - P_0

    @cached_property
    def rf_step(self) -> int:
        """Returns the step size (stride) between 2 receptive fields.

        Returns:
            int: step size/stride between 2 receptive fields.
        """
        assert (
            abs(self.rf_start_i(0) - self.rf_start_i(1))
            == abs(self.rf_end_i(0) - self.rf_end_i(1))
            == abs(self.rf_center_i(0) - self.rf_center_i(1))
        )
        return abs(self.rf_start_i(0) - self.rf_start_i(1))

    def n_windows(self, chunk_duration_f: int, strict: bool = True) -> int:
        """Compute the number of sliding windows over an audio chunk.
        Args:
            chunk_duration_f (int): Duration of the audio chunk in frames.
            strict (bool):
                If True (default), only count fully contained windows.
                If False, also count partial windows at the end.

        Returns:
            int: Number of valid windows. Returns 0 if the chunk is shorter than
                the window size in strict mode.
        """
        if chunk_duration_f <= 0:
            return 0

        if strict:
            if chunk_duration_f < self.rf_size:
                return 0
            # (L-W): Number of positions the window can slide from its starting point
            return (chunk_duration_f - self.rf_size) // self.rf_step + 1

        return ceil(chunk_duration_f / self.rf_step)
