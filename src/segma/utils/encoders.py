from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class LabelEncoder(ABC):
    @property
    @abstractmethod
    def labels(self) -> tuple[tuple[str, ...] | str, ...]:
        """Return encoded labels"""
        pass

    @property
    @abstractmethod
    def base_labels(self) -> tuple[str, ...]:
        """Return base labels that are encoded"""
        pass

    @abstractmethod
    def transform(self, label) -> int:
        pass

    @abstractmethod
    def inv_transform(self, i: int) -> str | tuple[str, ...]:
        pass

    @abstractmethod
    def one_hot(self, labels: Iterable[str] | str) -> np.ndarray:
        pass

    @abstractmethod
    def i_to_one_hot(self, i: int) -> np.ndarray:
        pass

    def __call__(self, labels: str | tuple[str, ...] = ()) -> int:
        return self.transform(labels)

    def __len__(self) -> int:
        """Returns the size of the generated powerset classes.

        Returns:
            int: _description_
        """
        raise NotImplementedError

    def __contains__(self, label: str | tuple[str, ...] | list[str]) -> bool:
        raise NotImplementedError


class MultiLabelEncoder(LabelEncoder):
    """Simmple encoder that maps a label to an integer."""

    def __init__(self, labels: list[str] | tuple[str, ...]) -> None:
        self._labels = labels
        self.n_labels = len(labels)

        self.map = {label: i for i, label in enumerate(labels)}
        self.rev_map = {i: label for label, i in self.map.items()}

    @property
    def labels(self) -> tuple[str, ...]:
        """Return encoded labels"""
        return tuple(self.map.keys())

    @property
    def base_labels(self) -> tuple[str, ...]:
        """Return base labels that are encoded"""
        return tuple(self._labels)

    def transform(self, label) -> int:
        """Transforms a label or a set of labels into its corresponding integer value
        representing the position of the label in the `y` one-hot vector used during training.
        """
        return self.map[label]

    def inv_transform(self, i: int) -> str:
        """takes an encoded label (integer) and returns the original labels as a tuple of strings."""
        if not (0 <= i < len(self)):
            raise ValueError(
                f"transformed index '{i}' is not assigned, only {len(self)} labels are available."
            )
        return self.rev_map[i]

    def one_hot(self, labels: Iterable[str] | str) -> np.ndarray:
        """Performs a transformation of the labels to its corresponding integer value and returns the correctly shaped corresponding one-hot vector.

        Args:
            labels (Iterable[str] | str): labels to transform.

        Returns:
            np.ndarray: one-hot vector of size `n_labels`.
        """
        labels = (labels,) if isinstance(labels, str) else labels
        idxs = [self.transform(label) for label in labels]
        base_one_hot = np.zeros(self.n_labels, dtype=int)
        base_one_hot[idxs] = 1
        return base_one_hot

    def i_to_one_hot(self, i: int) -> np.ndarray:
        # NOTE - not used
        labels = self.rev_map[i]
        return self.one_hot(labels)

    def __len__(self) -> int:
        """Returns the size of the generated powerset classes.

        Returns:
            int: _description_
        """
        return self.n_labels

    def __contains__(self, label: str | tuple[str, ...] | list[str]) -> bool:
        """check if one or multiple given raw labels are supported by the encoder."""
        if isinstance(label, list) or isinstance(label, tuple):
            raise ValueError(
                "Collections not supported, only single item membership makes sense"
            )
        return label in self.labels
