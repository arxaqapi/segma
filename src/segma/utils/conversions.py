def second_to_millisecond(s: float) -> float:
    return s * 1e3


def millisecond_to_second(ms: float) -> float:
    return ms / 1e3


def ms_to_s(ms: float) -> float:
    return millisecond_to_second(ms=ms)


def s_to_ms(s: float) -> float:
    return second_to_millisecond(s=s)


def seconds_to_frames(s: float, sample_rate: int = 16_000) -> int:
    return int(s * sample_rate)


def milliseconds_to_frames(ms: float, sample_rate: int = 16_000) -> float:
    return seconds_to_frames(ms_to_s(ms), sample_rate=sample_rate)


def frames_to_seconds(f: int, sample_rate: int = 16_000) -> float:
    return f / sample_rate


def frames_to_milliseconds(f: int, sample_rate: int = 16_000) -> float:
    return f / (sample_rate / 1e3)


def s_to_hms(s: float, rjust: int | None = 4) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    rs = int(s % 60)
    if rjust is not None:
        return f"{str(h).rjust(4)}h {str(m).rjust(4)}m {str(rs).rjust(4)}s"
    return f"{h}h {m}m {rs}s"
