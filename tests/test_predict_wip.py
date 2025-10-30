from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
from scipy.io import wavfile

from segma.models.base import ConvolutionSettings
from segma.predict_wip import (
    audio_slicer,
    generic_batched_windows,
    next_batch_start_i,
)


class DummyWavLMModel:
    def __init__(self):
        self.conv_settings = ConvolutionSettings(
            kernels=(10, 3, 3, 3, 3, 2, 2),
            strides=(5, 2, 2, 2, 2, 2, 2),
            paddings=(0,) * 7,
        )
        self.model = torch.nn.Sequential(
            *[
                torch.nn.Conv1d(1, 1, ks, s, 0)  # , bias=False)
                for ks, s in zip(self.conv_settings.kernels, self.conv_settings.strides)
            ]
        )

    def __call__(self, x: torch.Tensor) -> None:
        if len(x.shape) != 3:
            raise ValueError(
                f"Input dimensions should be (batch_size, n_channels, features), got `{x.shape}`"
            )
        with torch.no_grad():
            return self.model(x)


class DummyWhisperModel:
    def __init__(self):
        self.conv_settings = ConvolutionSettings(
            kernels=(400, 3, 3), strides=(160, 1, 2), paddings=(200, 1, 1)
        )
        self.model = torch.nn.Sequential(
            *[
                torch.nn.Conv1d(1, 1, ks, s, 0)  # , bias=False)
                for ks, s in zip(self.conv_settings.kernels, self.conv_settings.strides)
            ]
        )

    def __call__(self, x: torch.Tensor) -> None:
        if len(x.shape) != 3:
            raise ValueError(
                f"Input dimensions should be (batch_size, n_channels, features), got `{x.shape}`"
            )
        with torch.no_grad():
            return self.model(x)


SR = 16_000


@pytest.fixture
def _generate_audio():
    audio_path = Path("tests/sample/dummy.wav")

    dur_min = 3
    audio = np.arange(dur_min * 60 * SR, dtype=np.float32)[None, :]
    wavfile.write(audio_path, SR, audio.T)
    yield audio, audio_path
    audio_path.unlink(missing_ok=True)


def test_generic_batched_windows(_generate_audio):
    # test the return windows match the ones return by unfold
    audio, audio_path = _generate_audio
    assert audio.shape[-1] == 2880000

    chunk_duration_f = int(4 * SR)
    batch_size = 8
    model = DummyWavLMModel()

    batches = generic_batched_windows(
        torchaudio.info(audio_path).num_frames,
        partial(audio_slicer, audio_path=audio_path),
        step=next_batch_start_i(chunk_duration_f, model.conv_settings),
        size=chunk_duration_f,
        n=batch_size,
    )

    indices = []
    for batch in batches:
        for element in batch:
            indices.append(element.squeeze()[0].item())

    audio_t = torchaudio.load(audio_path)[0]
    folds = audio_t.squeeze().unfold(
        dimension=0,
        size=chunk_duration_f,
        step=next_batch_start_i(chunk_duration_f, model.conv_settings),
    )
    folds = [fold[0].item() for fold in folds]
    folds.append(folds[-1] + next_batch_start_i(chunk_duration_f, model.conv_settings))

    assert folds == indices


def test_generic_batched_windows_whisper_clipping(_generate_audio):
    audio, audio_path = _generate_audio
    assert audio.shape[-1] == 2880000

    chunk_duration_f = int(4 * SR)
    batch_size = 8
    model = DummyWhisperModel()

    batches = generic_batched_windows(
        torchaudio.info(audio_path).num_frames,
        partial(audio_slicer, audio_path=audio_path),
        start=model.conv_settings.rf_start_i(0),
        step=next_batch_start_i(chunk_duration_f, model.conv_settings),
        size=chunk_duration_f,
        n=batch_size,
    )
    batches = list(batches)

    assert batches[0][0][0][:520].sum() == 0.0
