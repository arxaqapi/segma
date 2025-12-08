from dataclasses import dataclass
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from segma.utils.conversions import frames_to_seconds


@dataclass
class AudioInfo:
    sample_rate: int
    n_samples: int
    n_channels: int


def get_audio_info(audio_p: Path) -> AudioInfo:
    """Returns number of frames and sample rate of the audio"""
    decoder = AudioDecoder(audio_p.resolve())
    decoder.metadata.duration
    return AudioInfo(
        n_samples=int(
            decoder.metadata.duration_seconds_from_header * decoder.metadata.sample_rate
        ),
        sample_rate=decoder.metadata.sample_rate,
        n_channels=decoder.metadata.num_channels
    )


def get_samples_in_range(audio_p: Path, start_f: int, duration_f: int) -> torch.Tensor:
    decoder = AudioDecoder(audio_p.resolve())
    return decoder.get_samples_played_in_range(
        start_seconds=frames_to_seconds(start_f),
        stop_seconds=frames_to_seconds(start_f + duration_f),
    ).data


def get_all_samples(audio_p: Path) -> torch.Tensor:
    decoder = AudioDecoder(audio_p.resolve())
    return decoder.get_all_samples().data


def write_data_to_disk(
    data: torch.Tensor, output_file: Path, sample_rate: int = 16_000
) -> None:
    AudioEncoder(data, sample_rate=sample_rate).to_file(output_file.with_suffix(".wav"))
