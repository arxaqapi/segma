from math import ceil
from pathlib import Path
from typing import Callable, Generator

import lightning as pl
import numpy as np
import torch
import torchaudio
from interlap import InterLap
from torch.utils.data import DataLoader, IterableDataset

from segma.config import Config
from segma.data.file_dataset import DatasetSubset, SegmaFileDataset
from segma.models.base import ConvolutionSettings
from segma.utils.conversions import (
    frames_to_seconds,
    seconds_to_frames,
)
from segma.utils.encoders import LabelEncoder


class DataLoaderError(Exception): ...


class SegmentationDataLoader(pl.LightningDataModule):
    """`SegmentationDataLoader` is a `pl.LightningDataModule` subclass that loads all required informations about the dataset
    and returns `AudioSegmentationDataset` (which are `IterableDataset`s) for training and validation.

    On initialization, the `SegmentationDataLoader` loads all uris,
    retrieves and store the total length of the corresponding audios and annotated duration,
    and createst `Interlap`objects per uri.
    """

    def __init__(
        self,
        dataset: SegmaFileDataset,
        label_encoder: LabelEncoder,
        config: Config,
        conv_settings: ConvolutionSettings,
        audio_preparation_hook: Callable | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.label_encoder = label_encoder
        self.config = config
        self.conv_settings = conv_settings
        self.audio_preparation_hook = audio_preparation_hook

        self.rng = np.random.default_rng()

        # NOTE - load dataset
        if not dataset.is_loaded():
            dataset.load()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=AudioSegmentationDataset(
                subset=self.dataset.train,
                config=self.config,
                conv_settings=self.conv_settings,
                label_encoder=self.label_encoder,
                audio_preparation_hook=self.audio_preparation_hook,
            ),
            batch_size=self.config.train.batch_size,
            drop_last=True,
            num_workers=self.config.train.dataloader.num_workers,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=AudioSegmentationDataset(
                subset=self.dataset.val,
                config=self.config,
                conv_settings=self.conv_settings,
                label_encoder=self.label_encoder,
                audio_preparation_hook=self.audio_preparation_hook,
            ),
            batch_size=self.config.train.batch_size,
            drop_last=True,
            num_workers=self.config.train.dataloader.num_workers,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
        )


class AudioSegmentationDataset(IterableDataset):
    """`AudioSegmentationDataset` is an `IterableDataset` that yields short audio segments and corresponding labels
    for supervised training. Audio is sampled randomly from full recordings with corresponding annotations.
    """

    def __init__(
        self,
        subset: DatasetSubset,
        config: Config,
        conv_settings: ConvolutionSettings,
        label_encoder: LabelEncoder,
        audio_preparation_hook: Callable | None = None,
    ) -> None:
        self.uris = subset.uris
        self.durations = subset.durations
        self.annotations = subset.interlaps

        self.config = config
        self.conv_settings = conv_settings
        self.label_encoder = label_encoder
        self.audio_preparation_hook = audio_preparation_hook

        self.windows = generate_frames(
            conv_settings=self.conv_settings,
            sample_rate=config.audio.sample_rate,
            chunk_duration_s=config.audio.chunk_duration_s,
            # NOTE - strict_frames: True is pyannet - False if whisper
            strict=config.audio.strict_frames,
        )

        assert len(self.uris) == self.durations.shape[0], (
            "Mismatch between URIs and durations."
        )

    def __iter__(self) -> Generator[dict[str, np.ndarray | torch.Tensor], None, None]:
        """Randomly samples an 'self.config.audio.chunk_duration_s' second audio sample with its corresponding labels.
        The labels correspond to overlapping frames with 'displacement' settings obtained from the models `ConvolutionSettings`.

        Yields:
            Generator[dict[str, np.ndarray]]: Yields a dict with content:

            - **x** (np.ndarray) -- Preprocessed waveform
            - **y** (np.ndarray) -- array of encoded target vectors. (d, n_labels)
        """
        w_info = torch.utils.data.get_worker_info()
        # Ensures that each worker has a separate seed
        rng = np.random.default_rng(seed=w_info.seed if w_info else None)
        durations_f = seconds_to_frames(self.config.audio.chunk_duration_s)
        while True:
            # NOTE - 1. Sample a file depending on its annotated or audio duration
            # audio_duration_f, annotated_duration_f
            weights = (
                self.durations["audio_duration_f"]
                / self.durations["audio_duration_f"].sum()
            )
            uri_i = rng.choice(len(self.uris), p=weights)

            # NOTE - 2. sample a start index between 0 and (audio.len - duration)
            start_index_f = int(
                rng.integers(
                    low=0,
                    high=self.durations["audio_duration_f"][uri_i] - durations_f,
                )
            )
            # NOTE - 3. {'x': cropped audio from [start_idx: start_idx + duration]
            #     'y': overlaping frames corresponding to the model output [[...n-labels], ...] }
            # (32000)
            audio_path = Path(self.config.data.dataset_path) / "wav" / self.uris[uri_i]
            waveform = self.load_audio(
                audio_file_p=audio_path.with_suffix(".wav"),
                start_f=start_index_f,
                duration_f=durations_f,
            )

            # if self.classify_sequence:
            #    y_target = windows_to_targets(
            #        np.asarray([[0,9216]]), self.label_encoder, self.annotations[uri_i]
            #    )
            # else :
            # NOTE - 4. generate corresponding sliding window 'y' vector and get labels
            windows = self.windows + start_index_f
            y_target = windows_to_targets(
                windows, self.label_encoder, self.annotations[uri_i]
            )

            # NOTE - 5. Optional preprocessing (e.g., STFT, Mel, normalization): (32000) to (1, 80, 3000)
            if self.audio_preparation_hook is not None:
                waveform = self.audio_preparation_hook(waveform)

            # x: (_, waveforms), y: (_, n_windows, n_labels)
            yield {"x": waveform.squeeze(), "y": y_target}

    def load_audio(
        self, audio_file_p: Path, start_f: int, duration_f: int
    ) -> torch.Tensor:
        """Loads a section of an audio file starting from frame `start_f` and spanning `duration_f` frames.
        The loaded audio is downmixed to mono if it has multiple channels.

        Args:
            audio_file_p (Path): Path to the audio file to load.
            start_f (int): Start frame to begin loading from.
            duration_f (int): Number of frames to load. Must match the expected chunk size from config.

        Raises:
            ValueError: If `duration_f` does not match the expected chunk duration in frames.

        Returns:
            torch.Tensor: A 1D tensor containing the mono audio waveform segment.
        """
        n_expected_frames = int(
            self.config.audio.chunk_duration_s * self.config.audio.sample_rate
        )
        if duration_f != n_expected_frames:
            raise ValueError(
                f"Error in `AudioSegmentationDataset.load_audio()`: `{duration_f=}` does not match expected `{n_expected_frames}` frames."
            )
        audio_t, _sr = torchaudio.load(
            audio_file_p.resolve(),
            frame_offset=start_f,
            num_frames=duration_f,
            # TODO - fix this with ffmpeg
            backend="soundfile",
        )
        # Downmix to mono if necessary
        if audio_t.shape[0] > 1:
            audio_t = audio_t.mean(dim=0)

        return audio_t.squeeze()

    def __len__(self) -> int:
        """Estimate the number of audio chunks sampled in one epoch.

        Since this is an `IterableDataset`, `__len__` provides a hint to the training loop
        about how many samples will be yielded during one epoch. This is especially useful
        for progress bars, checkpointing, or early stopping criteria.

        The number is computed based on the total available audio duration (in seconds),
        divided by the configured chunk duration (e.g., 2 seconds), and scaled by a
        `dataset_multiplier` from the config. To avoid epochs with too few samples,
        the result is also forced to be at least as large as the batch size.

        Note:
            This length does not represent the number of unique audio segments,
            but rather how many *samples* will be drawn per epoch, potentially
            with replacement or overlap.

        Returns:
            int: Estimated number of training samples (chunks) drawn in one epoch.
        """
        # audio_duration_f, annotated_duration_f
        total_annotated_duration_s = frames_to_seconds(
            self.durations["audio_duration_f"].sum()
        )
        return int(
            self.config.data.dataset_multiplier
            * max(
                ceil(total_annotated_duration_s / self.config.audio.chunk_duration_s),
                self.config.train.batch_size,
            )
        )


"""
    Given a models receptive field settings and a `chunk_duration_s`,
    compute the amount of possible frames to generate and generate these frames.
    (to be offsetted)
"""


def generate_frames(
    conv_settings: ConvolutionSettings,
    sample_rate: int,
    chunk_duration_s: float = 2.0,
    strict: bool = True,
) -> np.ndarray:
    """Generate the frame ranges (start and end indices) for each receptive field window
    over a given audio chunk.

    Args:
        conv_settings (ConvolutionSettings): Model-specific convolution parameters.
        sample_rate (int): Sample rate of the audio (e.g. 16000 Hz).
        chunk_duration_s (float, optional): Duration of the audio chunk in seconds. Defaults to 2.0.
        strict (bool, optional):
            If True, include only fully-contained windows.
            If False, allow partial windows that may exceed the chunk. Defaults to True.

    Returns:
        np.ndarray: An array of shape (n_windows, 2), where each row is [start_frame, end_frame].
    """
    # should be 32_000 for 2s @ 16khz
    # should be 96_000 for 6s @ 16khz
    chunk_duration_f = int(seconds_to_frames(chunk_duration_s, sample_rate))

    # if strict, each window will have the exact same size `rf_size(...)`,
    # else allow shorter frames that are then clipped
    # 352 with pyannet and cds=6
    n_windows = conv_settings.n_windows(
        chunk_duration_f=chunk_duration_f, strict=strict
    )
    windows = [
        [conv_settings.rf_start_i(i), conv_settings.rf_end_i(i)]
        for i in range(n_windows)
    ]
    # REVIEW - add -1 to chunk_duration_s ??
    return np.array(windows).clip(0, chunk_duration_f)


def windows_to_targets(
    windows: np.ndarray, label_encoder: LabelEncoder, labels: InterLap
) -> np.ndarray:
    """takes the windows (offseted correctly), a label encoder and the list of interlaps
    to generate the array of one-hot target vectors.
    Frames that have no class, are given the default one at index position '0'.p
    """

    y_target = []
    for w in windows:
        start, end = w
        intersecting_labels = {label for _, _, label in labels.find((start, end))}

        # NOTE - given a set of labels, return the one_hot representation
        y_target.append(label_encoder.one_hot(intersecting_labels))

    return np.array(y_target, dtype=np.float32)
