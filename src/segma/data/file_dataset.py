from functools import reduce
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import torchaudio
from interlap import InterLap

from segma.annotation import AudioAnnotation
from segma.utils.conversions import frames_to_seconds, seconds_to_frames


class URISubsetLeakageError(Exception):
    """Error raised when there is data leakage between the different defined subsets."""


def load_uris(file_p: Path) -> list[str]:
    """Loads a list of URIs from a given text file.

    Args:
        file_p (Path): Path to the file containing one URI per line.

    Returns:
        list[str]: A list of URIs as strings.

    Example:
        Contents of the file pointed to by `file_p`:
            ```
            # file_p content
            uri_001
            uri_002
            uri_003
            ```
    """
    with file_p.open("r") as f:
        uris = [line.strip() for line in f.readlines()]
    return uris


def load_annotations(aa_file_p: Path) -> list[AudioAnnotation]:
    """Loads audio annotations from a file.

    Args:
        aa_file_p (Path): Path to the file containing audio annotations.

    Returns:
        list[AudioAnnotation]: A list of parsed `AudioAnnotation` objects.
    """
    with aa_file_p.open("r") as f:
        annotations = [AudioAnnotation.read_line(line) for line in f.readlines()]
    return annotations


def total_annotation_duration_f(
    annotations: list[AudioAnnotation], sample_rate: int
) -> int:
    """Computed the total annotated duration in number of frames of a list of `AudioAnnotation` objects.

    Args:
        annotations (list[AudioAnnotation]): List of `AudioAnnotation` objects.
        sample_rate (int): Sample rate of the audio, used to convert to the right amount of frames

    Returns:
        float: Total duration in ms of all annotated segments.
    """
    return seconds_to_frames(
        reduce(lambda b, e: b + e.duration_s, annotations, 0.0), sample_rate=sample_rate
    )


def filter_annotations(
    annotations: list[AudioAnnotation],
    covered_labels: tuple[str, ...] | list[str] | set[str],
) -> list[AudioAnnotation]:
    """Filters a list of audio annotation by removing labels that are not in `covered_labels`.

    Args:
        annotations (list[AudioAnnotation]): A list of `AudioAnnotation` objects.
        covered_labels (tuple[str, ...]): tuple of labels to keep.

    Returns:
        list[AudioAnnotation]: A filtered list of `AudioAnnotation` objects.
    """
    return [annot for annot in annotations if annot.label in covered_labels]


def create_interlap_from_annotation(annotations: list[AudioAnnotation]):
    """Given a list of `AudioAnnotation`, create an `Interlap` object using the frame information

    Args:
        annotations (list[AudioAnnotation]): _description_

    Returns:
        _type_: _description_
    """
    # self.label_encoder(annot.label),
    return InterLap(
        [(annot.start_time_f, annot.end_time_f, annot.label) for annot in annotations]
    )


# REVIEW - If we want to oversample, store annotated duration per label for each file (optional)
class SegmaFileDataset:
    """
    Format of the dataset:
    ```
    dataset_name/
    ├── aa/
    │   └── 0000.aa
    ├── rttm/
    │   └── 0000.rttm
    ├── uem/ (optional)
    │   └── 0000.uem
    ├── wav/
    │   └── 0000.wav
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── exclude.txt (optional)
    ```

    `train.txt`, `val.txt`, `test.txt` and `exclude.txt` contain a list of uris separated with newlines.
    """

    SUBSET_NAMES = ("train", "val", "test")

    def __init__(
        self,
        base_p: Path | str,
        classes: list[str],
        chunk_duration_s: float,
        sample_rate: int = 16_000,
    ) -> None:
        self.base_p = Path(base_p)
        self.classes = classes
        self.chunk_duration_s = chunk_duration_s
        self.sample_rate = sample_rate

        self.removed_uris: dict[Literal["exclude.txt", "invalid"], set[str]] = {}
        self.subset_to_uris = self.load_all_uris()

        # Call `.load()` to populate the next 2 variables
        self.subds_to_durations: None | dict[str, np.ndarray] = None
        self.subds_to_interlaps: None | dict[str, list[InterLap]] = None

    def check_for_data_leakage(self, subset_to_uris: dict[str, set[str]]):
        for k1, k2 in combinations(self.SUBSET_NAMES, 2):
            overlap = subset_to_uris[k1] & subset_to_uris[k2]
            if overlap:
                raise URISubsetLeakageError(
                    f"Subset {k1} and {k2} are overlaping, which can be data leakage.\nOverlapping uris are: '{overlap=}'"
                )

    def load_all_uris(self):
        """For each subset defined in `SUBSET_NAMES`, load all uris and filter out uris in `exclude.txt`"""
        subset_to_uris: dict[str, set[str]] = {}
        for subset in self.SUBSET_NAMES:
            uri_list_p = (self.base_p / subset).with_suffix(".txt")
            # TODO - check for uri deduplication in list of uris
            subset_to_uris[subset] = set(
                load_uris(uri_list_p) if uri_list_p.exists() else []
            )
        # NOTE - handle exclusion file
        exclude_p = self.base_p / "exclude.txt"
        if exclude_p.exists():
            uris_to_remove = set(load_uris(exclude_p))
            subset_to_uris = {
                subset: set(filter(lambda e: e not in uris_to_remove, uris))
                for subset, uris in subset_to_uris.items()
            }
            self.removed_uris["exclude.txt"] = uris_to_remove
        self.check_for_data_leakage(subset_to_uris)
        return subset_to_uris

    def load(self) -> None:
        """Loads all annotation informations about the dataset,
        retrieves and store the total length of the corresponding audios and annotated duration,
        and createst `Interlap`objects per uri.

        Raises:
            ValueError: Raises if the datasets are empty after processing
        """
        # ~70h max per audio
        _durations_t = np.dtype(
            [("audio_duration_f", np.uint32), ("annotated_duration_f", np.uint32)]
        )
        subds_to_durations: dict[str, np.ndarray] = {}
        subds_to_interlaps: dict[str, list[InterLap]] = {
            subset: [] for subset in self.SUBSET_NAMES
        }
        uris_to_remove: set[str] = set()
        for subset in self.SUBSET_NAMES:
            durations: list[tuple[int, int]] = []
            for uri in self.subset_to_uris[subset]:
                uri_path = (self.wav_p / uri).with_suffix(".wav").resolve()
                info = torchaudio.info(uri=uri_path)
                # NOTE - check that the audio is valid
                if not self._validate_uri(info.num_frames, info.sample_rate):
                    uris_to_remove.add(uri)
                    continue

                annotations = load_annotations((self.aa_p / uri).with_suffix(".aa"))
                # NOTE - Only labels covered in the config file are kept.
                annotations = filter_annotations(annotations, self.classes)

                subds_to_interlaps[subset].append(
                    create_interlap_from_annotation(annotations)
                )

                durations.append(
                    (
                        info.num_frames,
                        total_annotation_duration_f(annotations, self.sample_rate),
                    )
                )
            # Once all files have been treated, create an efficient np.array
            subds_to_durations[subset] = np.array(durations, dtype=_durations_t)

        # NOTE remove if not valid
        self.removed_uris["invalid"] = uris_to_remove
        for subset in self.SUBSET_NAMES:
            self.subset_to_uris[subset] -= uris_to_remove

        for subset, uris in self.subset_to_uris.items():
            if len(uris) == 0:
                raise ValueError(
                    f"subset '{subset}' is empty after removing all audio instances with duration < {self.chunk_duration_s} s and all audios/segments with invalid labels.\n"
                )

        self.subds_to_durations = subds_to_durations
        self.subds_to_interlaps = subds_to_interlaps

        # TODO - implement caching
        # try:
        #     self.load_cache()
        # except: pass

    def _validate_uri(self, num_frames: int, sample_rate: int) -> bool:
        """valide the audio file, if it is not valid:
        - size needs to be larger than chunk_duration_s
        - sample_rate need to match
        """
        return (
            frames_to_seconds(num_frames, sample_rate) >= self.chunk_duration_s
            and sample_rate == self.sample_rate
        )

    def load_cache(self):
        """Load the cached dataset informations."""
        raise NotImplementedError

    def save_cache(self):
        """Save the computed dataset informations to disk."""
        raise NotImplementedError

    @property
    def aa_p(self) -> Path:
        return self.base_p / "aa"

    @property
    def rttm_p(self) -> Path:
        return self.base_p / "rttm"

    @property
    def uem_p(self) -> Path:
        return self.base_p / "uem"

    @property
    def wav_p(self) -> Path:
        return self.base_p / "wav"
