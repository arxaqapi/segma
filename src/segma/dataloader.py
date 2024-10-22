"""Data loading utilies"""

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from math import ceil, floor
from pathlib import Path
from typing import Callable

import lightning as pl
import numpy as np
import torch
import torchaudio
from interlap import InterLap
from torch.utils.data import DataLoader, IterableDataset

from segma.utils.encoders import LabelEncoder
from segma.utils.receptive_fields import rf_center_i, rf_end_i, rf_size, rf_start_i

from .annotation import AudioAnnotation
from .models import ConvolutionSettings
from .utils.conversions import (
    frames_to_seconds,
    milliseconds_to_frames,
    seconds_to_frames,
)


def load_uris(file_p: Path) -> list[str]:
    """loads a list of uris"""
    with file_p.open("r") as f:
        uris = [line.strip() for line in f.readlines()]
    return uris


def load_annotations(aa_file_p: Path) -> list[AudioAnnotation]:
    with aa_file_p.open("r") as f:
        annotations = [AudioAnnotation.read_line(line) for line in f.readlines()]
    return annotations


def filter_annotations(
    annotations: list[AudioAnnotation], covered_labels: tuple[str, ...]
):
    return [annot for annot in annotations if annot.label in covered_labels]


def total_annotation_duration(annotations: list[AudioAnnotation]) -> float:
    return reduce(lambda b, e: b + e.duration_ms, annotations, 0)


@dataclass(frozen=True)
class Config:
    conv_settings: ConvolutionSettings
    labels: tuple[str, ...]
    sample_rate: int = 16_000
    chunk_duration_s: float = 2.0
    batch_size: int = 32
    num_workers: int = 4
    ds_path: Path = Path("data/debug")


class SegmentationDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        label_encoder: LabelEncoder,
        config: Config,
        audio_preparation_hook: Callable | None = None,
    ) -> None:
        super().__init__()
        self.label_encoder = label_encoder
        self.config = config
        self.audio_preparation_hook = audio_preparation_hook

        self.rng = np.random.default_rng()

        # NOTE - load train, val, test uris
        self.uris: dict[str, list[str]] = {
            subset: load_uris((config.ds_path / subset).with_suffix(".txt"))
            for subset in ("train", "val", "test")
        }

        # NOTE for each subset, get and store audio duration and annotated duration (as number of frames)
        _durations_t = np.dtype(
            [("audio_duration_f", np.int32), ("annotated_duration_f", np.int32)]
        )
        self.subds_to_durations: dict[str, np.ndarray] = {}
        """mapping from subset to arrays of tuples containing the audio duration and total annotated duration, as number of frames"""
        _subds_to_annotations: dict[str, list[list[AudioAnnotation]]] = defaultdict(
            list
        )
        uris_to_remove: list[tuple[str, str]] = []
        for subset in ("train", "val", "test"):
            duration_l = []
            for uri in self.uris[subset]:
                # total audio duration in number of frames sampled at self.sample_rate
                info = torchaudio.info(
                    uri=(config.ds_path / "wav" / uri)
                    .with_suffix(".wav")
                    .readlink()
                    .absolute()
                )
                annotations = load_annotations(
                    (config.ds_path / "aa" / uri).with_suffix(".aa")
                )

                if not self._validate_uri(
                    num_frames=info.num_frames,
                    sample_rate=info.sample_rate,
                    annotations=annotations,
                ):
                    uris_to_remove.append((subset, uri))
                    continue

                # REVIEW - annotation are stored here and reused later, such that filtering has to happen only once.
                annotations = filter_annotations(
                    annotations, covered_labels=config.labels
                )
                _subds_to_annotations[subset].append(annotations)

                duration_l.append(
                    (info.num_frames, total_annotation_duration(annotations))
                )

            # NOTE - make efficient np.array
            self.subds_to_durations[subset] = np.array(duration_l, dtype=_durations_t)

        # NOTE - remove all uris where the audio is shorter that the given duration
        for subset, uri_to_remove in uris_to_remove:
            self.uris[subset].remove(uri_to_remove)

        # NOTE - load all annotations as mapping from subset to list of Interlap object (that allows fast query for `y` vector construction)
        self.subds_to_annotations: dict[str, list[InterLap]] = defaultdict(list)
        for subset in ("train", "val", "test"):
            for uri, annotations in zip(
                self.uris[subset], _subds_to_annotations[subset]
            ):
                # for each uri, all annotations are retrieved and transformed into an interlap object
                self.subds_to_annotations[subset].append(
                    # REVIEW - Interlap add does work with int only
                    InterLap(
                        [
                            (
                                int(
                                    milliseconds_to_frames(
                                        annot.start_time_ms,
                                        sample_rate=config.sample_rate,
                                    )
                                ),
                                int(
                                    milliseconds_to_frames(
                                        annot.end_time_ms,
                                        sample_rate=config.sample_rate,
                                    )
                                ),
                                # self.label_encoder(annot.label),
                                annot.label,
                            )
                            for annot in annotations
                        ]
                    )
                )

    def _validate_uri(
        self, num_frames, sample_rate, annotations: list[AudioAnnotation]
    ) -> bool:
        """valide the audio file, if it is not valid
        (sample_rate does not match, label do not match, shorter than chunk_duration_s),
        return False
        """
        # TODO - handle the case when audio_duraton == chunk_duration_s -> pick index 0
        # TODO - raise warnings
        if frames_to_seconds(num_frames, sample_rate) <= self.config.chunk_duration_s:
            return False

        if sample_rate != self.config.sample_rate:
            return False

        for annot in annotations:
            if annot.label not in self.label_encoder:
                return False
        return True

    def train_dataloader(self):
        return DataLoader(
            dataset=AudioSegmentationDataset(
                uris=self.uris["train"],
                durations=self.subds_to_durations["train"],
                annotations=self.subds_to_annotations["train"],
                config=self.config,
                label_encoder=self.label_encoder,
                audio_preparation_hook=self.audio_preparation_hook,
            ),
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=AudioSegmentationDataset(
                uris=self.uris["val"],
                durations=self.subds_to_durations["val"],
                annotations=self.subds_to_annotations["val"],
                config=self.config,
                label_encoder=self.label_encoder,
                audio_preparation_hook=self.audio_preparation_hook,
            ),
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
        )


class AudioSegmentationDataset(IterableDataset):
    def __init__(
        self,
        uris: list[str],
        durations: np.ndarray,
        annotations: list[InterLap],
        config: Config,
        label_encoder: LabelEncoder,
        audio_preparation_hook: Callable | None = None,
    ):
        self.uris = uris
        self.durations = durations
        self.annotations = annotations

        self.config = config
        self.label_encoder = label_encoder
        self.audio_preparation_hook = audio_preparation_hook

        self.windows = generate_frames(
            config.conv_settings,
            config.sample_rate,
            config.chunk_duration_s,
            strict=False,
        )

        assert len(uris) == durations.shape[0]

    def __iter__(self):
        w_info = torch.utils.data.get_worker_info()
        # ensures each worker has a separate seed
        rng = np.random.default_rng(seed=w_info.seed)

        durations_f = floor(seconds_to_frames(self.config.chunk_duration))
        while True:
            # NOTE 1. sample a file depending on its annotated or audio duration
            # audio_duration_f, annotated_duration_f
            uri_i = rng.choice(
                np.arange(len(self.uris)),
                p=self.durations["audio_duration_f"]
                / self.durations["audio_duration_f"].sum(),
            )

            # NOTE 2. sample a start index between 0 and (audio.len - duration)
            start_index_f = int(
                rng.integers(
                    low=0,
                    high=self.durations["audio_duration_f"][uri_i] - durations_f,
                )
            )

            # NOTE 3. {'x': cropped audio from [start_idx: start_idx + duration]
            #     'y': overlaping frames corresponding to the model output [[...n-labels], ...] }
            # (32000)
            waveform = self.load_audio(
                (self.config.ds_path / "wav" / self.uris[uri_i]).with_suffix(".wav"),
                start_f=start_index_f,
                duration_f=durations_f,
            )

            # NOTE 4. generate corresponding sliding window 'y' vector and get labels
            windows = self.windows + start_index_f
            y_target = windows_to_targets(
                windows, self.label_encoder, self.annotations[uri_i]
            )

            # NOTE - preparation hook
            if self.audio_preparation_hook is not None:
                # (1, 80, 3000)
                waveform = self.audio_preparation_hook(waveform)["input_features"]

            # x: (_, waveforms), y: (_, n_windows, n_labels)
            yield {"x": waveform.squeeze(), "y": y_target}

    def load_audio(self, audio_file_p: Path, start_f: int, duration_f: int):
        """loads only wanted segment from audio and downsamples it."""
        assert duration_f == 32_000
        audio_t, _sr = torchaudio.load(
            audio_file_p.resolve(), frame_offset=start_f, num_frames=duration_f
        )
        # downmix to mono
        if audio_t.shape[0] > 1:
            audio_t = audio_t.mean(dim=0)
        return audio_t.squeeze()

    def __len__(self):
        # total size of the dataset, signals a complete pass over the data
        # This will create multiple passes in the same epoch if batch_size is smaller that n (this value)
        # NOTE - we want to ensure that there is at least one batch, since incomplete batches are not kept
        # audio_duration_f, annotated_duration_f
        total_annotated_duration_s = frames_to_seconds(
            self.durations["audio_duration_f"].sum()
        )
        # REVIEW - file error with iterable dataset len that has to be an integer
        return int(
            max(
                ceil(total_annotated_duration_s / self.config.chunk_duration),
                self.config.batch_size,
            )
        )


def generate_frames(
    conv_settings: ConvolutionSettings,
    sample_rate: int,
    chunk_duration_s: float = 2.0,
    strict: bool = True,
) -> np.ndarray:
    """Given a models receptive field settings and a `chunk_duration_s`,
    compute the amount of possible frames to generate and generate these frames.
    (to be offsetted)
    """
    # should be 32_000 for 2s @ 16khz
    chunk_duration_f = int(seconds_to_frames(chunk_duration_s, sample_rate))

    # if strict, each window will have the exact same size `rf_size(...)`,
    # else allow shorter frames that are then clipped
    n_windows = conv_settings.n_windows(strict=strict, correct=True)
    wins = [
        [
            rf_start_i(i, conv_settings.strides, conv_settings.paddings),
            rf_end_i(
                i,
                conv_settings.kernels,
                conv_settings.strides,
                conv_settings.paddings,
            ),
        ]
        for i in range(n_windows)
    ]
    # REVIEW - add -1 to chunk_duration_s ??
    wins = np.array(wins).clip(0, chunk_duration_f)

    return wins


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
        intersecting_labels = set([label for _, _, label in labels.find((start, end))])

        y_target.append(label_encoder.one_hot(intersecting_labels))

    return np.array(y_target, dtype=np.float32)
