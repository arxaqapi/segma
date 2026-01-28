from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Self

import numpy as np
from interlap import InterLap

from segma.config import Config
from segma.data.utils import (
    create_interlap_from_annotation,
    filter_annotations,
    load_annotations,
    load_uris,
    total_annotation_duration_f,
)
from segma.utils.conversions import frames_to_seconds
from segma.utils.io import get_audio_info


class DatasetNotLoadedError(Exception): ...


class URISubsetLeakageError(Exception):
    """Error raised when there is data leakage between the different defined subsets."""


@dataclass
class DatasetSubset:
    uris: list
    durations: np.ndarray
    interlaps: list[InterLap]


# REVIEW - If we want to oversample, store annotated duration per label for each file (optional)
class SegmaFileDataset:
    """Loads a dataset in a multistep way, handling exclusion,
    file validation and caching of computed metrics.

    Format of the dataset:
    ```
    dataset_name/
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

    `train.txt`, `val.txt`, `test.txt` and `exclude.txt` contain a list of uris
    separated with newlines.
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
        if not self.base_p.exists():
            raise FileNotFoundError(
                f"Given path to the dataset is non existent. Got `{self.base_p}`."
            )
        self.classes = classes
        self.chunk_duration_s = chunk_duration_s
        self.sample_rate = sample_rate

        self.removed_uris: dict[
            Literal[
                "exclude.txt",
                "invalid",
                "duplicate.train",
                "duplicate.val",
                "duplicate.test",
            ],
            set[str],
        ] = {}
        self.subset_to_uris: dict[str, list[str]] = self.load_all_uris()

        # Call `.load()` to populate the next 2 variables
        self.subds_to_durations: None | dict[str, np.ndarray] = None
        self.subds_to_interlaps: None | dict[str, list[InterLap]] = None

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(
            config.data.dataset_path,
            config.data.classes,
            config.audio.chunk_duration_s,
            config.audio.sample_rate,
        )

    def check_for_data_leakage(self, subset_to_uris: dict[str, list[str]]) -> None:
        """Check that the uris sets do not intersect.
        Parirwise set intersection is checked such that they are empty.

        Args:
            subset_to_uris (dict[str, set[str]]): Dict that maps the subset name to the uris.

        Raises:
            URISubsetLeakageError: Raises when there is data leakage between two dataset subset.
        """
        for k1, k2 in combinations(self.SUBSET_NAMES, 2):
            overlap = set(subset_to_uris[k1]) & set(subset_to_uris[k2])
            if overlap:
                raise URISubsetLeakageError(
                    f"Subset {k1} and {k2} are overlaping, which can be data leakage.\nOverlapping uris are: '{overlap=}'"
                )

    def load_all_uris(self) -> dict[str, list[str]]:
        """For each subset defined in `SUBSET_NAMES`, load all uris
        and filter out uris present in `exclude.txt`.

        Returns:
            dict[str, set[str]]: _description_
        """
        subset_to_uris: dict[str, list[str]] = {}
        for subset in self.SUBSET_NAMES:
            uri_list_p = (self.base_p / subset).with_suffix(".txt")
            # NOTE - checks for uri deduplication in list of uris
            uri_list = load_uris(uri_list_p) if uri_list_p.exists() else []
            duplicates = [x for x, y in Counter(uri_list).items() if y > 1]
            if len(duplicates):
                self.removed_uris[f"duplicate.{subset}"] = duplicates
            subset_to_uris[subset] = uri_list
        # NOTE - handle exclusion file
        exclude_p = self.base_p / "exclude.txt"
        if exclude_p.exists():
            uris_to_remove = set(load_uris(exclude_p))
            subset_to_uris = {
                subset: list(filter(lambda e: e not in uris_to_remove, uris))
                for subset, uris in subset_to_uris.items()
            }
            self.removed_uris["exclude.txt"] = uris_to_remove

        self.check_for_data_leakage(subset_to_uris)
        return subset_to_uris

    def load(self) -> None:
        """Loads all annotation informations about the dataset,
        retrieves and store the total length of the corresponding audios and annotated duration,
        and createst `Interlap`objects per uri.
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
                info = get_audio_info(uri_path)
                # NOTE - check that the audio is valid
                if not self._validate_uri(info.n_samples, info.sample_rate):
                    uris_to_remove.add(uri)
                    continue

                annotations = load_annotations((self.rttm_p / uri).with_suffix(".rttm"))
                # NOTE - Only labels covered in the config file are kept.
                annotations = filter_annotations(annotations, self.classes)

                subds_to_interlaps[subset].append(
                    create_interlap_from_annotation(annotations)
                )

                durations.append(
                    (
                        info.n_samples,
                        total_annotation_duration_f(annotations, self.sample_rate),
                    )
                )
            # Once all files have been treated, create an efficient np.array
            subds_to_durations[subset] = np.array(durations, dtype=_durations_t)

        # NOTE remove if not valid
        self.removed_uris["invalid"] = uris_to_remove
        for subset in self.SUBSET_NAMES:
            self.subset_to_uris[subset] = list(
                filter(lambda e: e not in uris_to_remove, self.subset_to_uris[subset])
            )

        for subset, uris in self.subset_to_uris.items():
            if len(uris) == 0:
                raise ValueError(
                    f"subset '{subset}' is empty after removing all audio instances with duration < {self.chunk_duration_s} s and all audios/segments with invalid labels.\n"
                )
        print("Uris that have been removed", len(uris_to_remove))
        self.subds_to_durations = subds_to_durations
        self.subds_to_interlaps = subds_to_interlaps

    def _validate_uri(self, num_frames: int, sample_rate: int) -> bool:
        """Validate the audio file, checks that the size is bigger than `chunk_duration_s`
        and that the ample rate matches the one defined in `self.sample_rate`.
        - sample_rate need to match

        Args:
            num_frames (int): number of frames in the loaded audio.
            sample_rate (int): sample rate of the loaded audio.

        Returns:
            bool: Returns True if the audio verifies all conditions, else False.
        """
        return (
            frames_to_seconds(num_frames, sample_rate) >= self.chunk_duration_s
            and sample_rate == self.sample_rate
        )

    def is_loaded(self, raises: bool = False) -> bool:
        """Verifies that the dataset has been loaded.

        Args:
            raises (bool, optional): Behaviour can be set to raise instead of returning a boolean value. Defaults to False.

        Raises:
            DatasetNotLoadedError: Raised if `raises` set to True and the dataset has not been loaded.

        Returns:
            bool: Return True if the dataset has been loaded, else return False.
        """
        is_loaded = (
            self.subds_to_durations is not None and self.subds_to_interlaps is not None
        )
        if raises and not is_loaded:
            raise DatasetNotLoadedError
        return is_loaded

    @property
    def rttm_p(self) -> Path:
        return self.base_p / "rttm"

    @property
    def uem_p(self) -> Path:
        return self.base_p / "uem"

    @property
    def wav_p(self) -> Path:
        return self.base_p / "wav"

    @property
    def train(self) -> DatasetSubset:
        self.is_loaded(raises=True)
        return DatasetSubset(
            uris=self.subset_to_uris["train"],
            durations=self.subds_to_durations["train"],
            interlaps=self.subds_to_interlaps["train"],
        )

    @property
    def val(self) -> DatasetSubset:
        self.is_loaded(raises=True)
        return DatasetSubset(
            uris=self.subset_to_uris["val"],
            durations=self.subds_to_durations["val"],
            interlaps=self.subds_to_interlaps["val"],
        )

    @property
    def test(self) -> DatasetSubset:
        self.is_loaded(raises=True)
        return DatasetSubset(
            uris=self.subset_to_uris["test"],
            durations=self.subds_to_durations["test"],
            interlaps=self.subds_to_interlaps["test"],
        )
