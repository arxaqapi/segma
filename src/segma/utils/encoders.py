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


class MultiLabelEncoder(LabelEncoder):
    """Safe mapping from labels to integer values and the corresponding one-hot vectors for multi-label problems."""

    def __init__(self, labels: list[str] | tuple[str, ...]) -> None:
        self.map = {label: i for i, label in enumerate(labels)}

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.map.keys())

    def transform(self, label: str) -> int:
        """label to corresponding integer value."""
        if label not in self.map.keys():
            raise ValueError(
                f"{label} is not part of the supported labels of this LabelEncoder instance.\nOnly labels: '{list(self.map.keys())}' are supported."
            )
        return self.map[label]

    def inv_transform(self, i: int) -> tuple[str, ...]:
        raise NotImplementedError

    def one_hot(self, labels: Iterable[str] | str) -> np.ndarray:
        """Returns the one-hot representation of the label of list of labels given as input.

        Args:
            label (str | list[str]): the label.s to transform into the respective one-hot representation.

        Raises:
            ValueError: Fails if the label is not of type string or list of strings.

        Returns:
            np.ndarray: numpy array of size n.
        """
        _n = len(self.map)
        if isinstance(labels, Iterable):
            return (
                np.eye(_n, dtype=np.uint8)[[self.transform(lab) for lab in labels]]
                .sum(axis=0)
                .clip(0, 1)
            )
        elif isinstance(labels, str):
            return np.eye(_n, dtype=np.uint8)[self.transform(labels)]
        else:
            raise ValueError(
                f"input argument {labels=} has incorrect type '{type(labels)}', used str or list[str] as argument."
            )

    def i_to_one_hot(self, i: int) -> np.ndarray:
        raise NotImplementedError

    def inv_one_hot(self, one_hot: np.ndarray) -> tuple[str, ...]:
        """maps a one_hot vector to its corresponding labels"""
        if not len(one_hot.shape) == 1:
            raise NotImplementedError(
                "Only a single one-hot vector is supported at the moment for the inverse transform function."
            )
        if not one_hot.shape[-1] == len(self):
            raise ValueError(
                f"The input one-hot vector has size {one_hot.shape[-1]} wich is different than the expected '{len(self)}' encoded labels."
            )

        return tuple([label for label, i in self.map.items() if one_hot[i] == 1])

    one_hot_to_label = inv_one_hot

    def __call__(self, label: str) -> int:
        return self.transform(label)

    def __len__(self) -> int:
        return len(self.map)

    def __contains__(self, x) -> bool:
        return x in self.labels


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
