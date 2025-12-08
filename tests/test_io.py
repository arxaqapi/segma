from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from segma.utils.io import get_all_samples, get_audio_info, get_samples_in_range


@pytest.fixture
def _prepare_dummy_wav_files():
    # NOTE - Create test wav files
    out_p = Path("tests/sample/wav")
    out_p.mkdir(exist_ok=True, parents=True)
    sr = 16_000
    audio = np.zeros((1, 3 * 60 * sr), dtype=np.float32)

    wavfile.write(out_p / "00.wav", sr, audio.T)

    yield
    # NOTE - remove test wav files
    for wav in out_p.glob("*.wav"):
        wav.unlink()
    out_p.rmdir()


def test_load(_prepare_dummy_wav_files):
    for audio_p in Path("tests/sample/wav").glob("*.wav"):
        # NOTE - info
        info = get_audio_info(audio_p)
        assert info.n_channels == 1
        assert info.sample_rate == 16_000
        assert info.n_samples == (3 * 60 * 16_000)

        all_audio_t = get_all_samples(audio_p)
        assert all_audio_t.shape == (1, 3 * 60 * 16_000)

        audio_t = get_samples_in_range(audio_p, 0, 3 * 60 * 16_000)
        assert audio_t.shape == (1, 3 * 60 * 16_000)

        audio_t = get_samples_in_range(audio_p, 10_000, -1)
        assert audio_t.shape == (1, 3 * 60 * 16_000 - 10_000)
