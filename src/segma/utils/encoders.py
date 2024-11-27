from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations

import numpy as np


class LabelEncoder(ABC):
    @property
    @abstractmethod
    def labels(self) -> tuple[tuple[str, ...], ...]:
        pass

    @abstractmethod
    def transform(self, label) -> int:
        pass

    @abstractmethod
    def inv_transform(self, i: int) -> tuple[str, ...]:
        pass

    @abstractmethod
    def one_hot(self, labels: Iterable[str] | str) -> np.ndarray:
        pass

    @abstractmethod
    def i_to_one_hot(self, i: int) -> np.ndarray:
        pass


class PowersetMultiLabelEncoder(LabelEncoder):
    """Safe mapping from labels to integer values and the corresponding one-hot vectors for multi-label problems.
    Automatically transforms a multi-label problem into a multi-class one."""

    def __init__(self, labels: list[str] | tuple[str, ...]) -> None:
        self._labels = labels
        powerset = [()] + [
            e for i in range(len(labels)) for e in list(combinations(labels, i + 1))
        ]
        self.n_labels = len(powerset)
        self.map = {tuple(sorted(labels)): i for i, labels in enumerate(powerset)}
        self.rev_map = {i: labels for labels, i in self.map.items()}

    @property
    def labels(self) -> tuple[tuple[str, ...], ...]:
        """return all label combinations"""
        return tuple(self.map.keys())

    def transform(self, labels: Iterable[str] | str) -> int:
        """labels to integer value"""
        labels = (labels,) if isinstance(labels, str) else tuple(sorted(labels))
        return self.map[labels]

    def inv_transform(self, i: int) -> tuple[str, ...]:
        """takes an encoded label and returns the original label string."""
        if i >= len(self):
            raise ValueError(
                f"transformed index '{i}' is not assigned, only {len(self)} labels are available."
            )
        return self.rev_map[i]

    def one_hot(self, labels: Iterable[str] | str) -> np.ndarray:
        """labels to their one-hot representation"""
        return np.eye(self.n_labels, dtype=np.uint8)[self.transform(labels)]

    def i_to_one_hot(self, i: int) -> np.ndarray:
        """takes the encoded values of the encoder and returns the correponding one-hot vector."""
        labels = self.rev_map[i]
        return self.one_hot(labels)

    def __call__(self, labels: str | tuple[str, ...] = ()) -> int:
        return self.transform(labels)

    def __len__(self) -> int:
        return len(self.map)

    def __contains__(self, label: str | tuple[str, ...] | list[str]) -> bool:
        if isinstance(label, str):
            label = (label,)
        return tuple(sorted(label)) in self.labels
