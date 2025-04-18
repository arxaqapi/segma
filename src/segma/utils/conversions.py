import numpy as np


def second_to_millisecond(s: float | np.ndarray) -> float | np.ndarray:
    return s * 1e3


def millisecond_to_second(ms: float | np.ndarray) -> float | np.ndarray:
    return ms / 1e3


def ms_to_s(ms: float | np.ndarray) -> float | np.ndarray:
    return millisecond_to_second(ms=ms)


def s_to_ms(s: float | np.ndarray) -> float | np.ndarray:
    return second_to_millisecond(s=s)


def seconds_to_frames(
    s: float | np.ndarray, sample_rate: int = 16_000
) -> int | np.ndarray:
    return np.floor(s * sample_rate)


def milliseconds_to_frames(
    ms: float | np.ndarray, sample_rate: int = 16_000
) -> float | np.ndarray:
    return seconds_to_frames(ms_to_s(ms), sample_rate=sample_rate)


def frames_to_seconds(
    f: int | np.ndarray, sample_rate: int = 16_000
) -> float | np.ndarray:
    return f / sample_rate


def frames_to_milliseconds(
    f: int | np.ndarray, sample_rate: int = 16_000
) -> float | np.ndarray:
    return f / (sample_rate / 1e3)
