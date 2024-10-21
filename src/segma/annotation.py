from dataclasses import dataclass
from typing import Self

from .utils.conversions import (
    millisecond_to_second,
    second_to_millisecond,
)


@dataclass
class AudioAnnotation:
    uid: str
    start_time_ms: float
    """start_time in milliseconds"""
    duration_ms: float
    """duration in milliseconds"""
    label: str
    """associated segment label"""

    @classmethod
    def read_line(cls, line: str) -> Self:
        """parses a valid AudioAnnotation line as a string and outputs an AudioAnnotation instance."""
        uid, start_time, duration, label = line.strip().split(" ")
        return cls(uid, float(start_time), float(duration), label)

    @property
    def start_time_s(self) -> float:
        return millisecond_to_second(self.start_time_ms)

    @property
    def duration_s(self) -> float:
        return millisecond_to_second(self.duration_ms)

    @property
    def end_time_ms(self) -> float:
        return self.start_time_ms + self.duration_ms

    @property
    def end_time_s(self) -> float:
        return millisecond_to_second(self.end_time_ms)

    def write(self, n_digits: int = 6) -> str:
        return f"{self.uid} {round(self.start_time_ms, n_digits)} {round(self.duration_ms, n_digits)} {self.label}"

    def __str__(self) -> str:
        return f"Annot for '{self.uid}': from {round(self.start_time_s, 6)} s to {round(self.start_time_s + self.duration_s, 6)} | seg duration: {round(self.duration_s, 4)} | label: {self.label}"

    def __repr__(self) -> str:
        return self.write()

    def to_rttm(self) -> str:
        return " ".join(
            [
                "SPEAKER",
                self.uid,
                # "1",
                "<NA>",
                f"{round(self.start_time_s, 6)}",
                f"{round(self.duration_s, 6)}",
                "<NA> <NA>",
                self.label,
                "<NA> <NA>",
            ]
        )

    @classmethod
    def from_rttm(cls, line: str) -> Self:
        """parses a valid RTTM-formatted line as a string and outputs an AudioAnnotation instance."""
        fields = line.strip().split(" ")
        assert len(fields) == 10 or len(fields) == 9
        return cls(
            uid=fields[1],
            start_time_ms=second_to_millisecond(float(fields[3])),
            duration_ms=second_to_millisecond(float(fields[4])),
            label=fields[7],
        )
