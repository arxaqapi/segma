from dataclasses import dataclass
from typing import Self

from .utils.conversions import second_to_millisecond, seconds_to_frames


@dataclass
class AudioAnnotation:
    """
    Represents a labeled segment of audio data.

    Attributes:
        uid (str): Unique identifier for the audio file (URI).
        start_time_s (float): Start time of the segment in seconds.
        duration_s (float): Duration of the segment in seconds.
        label (str): Label associated with the segment.
    """

    uid: str
    start_time_s: float
    duration_s: float
    label: str
    PRECISION: int = 8

    @property
    def start_time_ms(self) -> float:
        return second_to_millisecond(self.start_time_s)

    @property
    def duration_ms(self) -> float:
        return second_to_millisecond(self.duration_s)

    @property
    def end_time_ms(self) -> float:
        return second_to_millisecond(self.end_time_s)

    @property
    def start_time_f(self) -> int:
        return seconds_to_frames(self.start_time_s)

    @property
    def duration_f(self) -> int:
        return seconds_to_frames(self.duration_s)

    @property
    def end_time_f(self) -> int:
        return seconds_to_frames(self.end_time_s)

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s

    def __str__(self) -> str:
        """Human-readable string representation of the annotation."""
        return f"Annot for '{self.uid}': from {round(self.start_time_s, self.PRECISION)} s to {round(self.end_time_s, self.PRECISION)} | seg duration: {round(self.duration_s, self.PRECISION)} | label: {self.label}"

    def __repr__(self) -> str:
        return f"{self.uid} {round(self.start_time_s, self.PRECISION)} {round(self.duration_s, self.PRECISION)} {self.label}"

    def to_rttm(self) -> str:
        """Convert the annotation into RTTM (Rich Transcription Time Marked) format.

        Returns:
            str: RTTM-formatted string for use with audio-diarization and segmentation tools.
        """
        return " ".join(
            [
                "SPEAKER",
                self.uid,
                # "1",
                "<NA>",
                f"{round(self.start_time_s, self.PRECISION)}",
                f"{round(self.duration_s, self.PRECISION)}",
                "<NA> <NA>",
                self.label,
                "<NA> <NA>",
            ]
        )

    @classmethod
    def from_rttm(cls, line: str) -> Self:
        """Parse an RTTM-formatted line into an AudioAnnotation instance.

        Args:
            line (str): RTTM-formatted string line.

        Returns:
            AudioAnnotation: Parsed annotation.
        """
        fields = line.strip().split(" ")
        assert len(fields) == 10 or len(fields) == 9
        return cls(
            uid=fields[1],
            start_time_s=float(fields[3]),
            duration_s=float(fields[4]),
            label=fields[7],
        )
