from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations

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


class PowersetMultiLabelEncoder(LabelEncoder):
    """Safe mapping from labels to integer values and the corresponding one-hot vectors for multi-label problems.
    Automatically transforms a multi-label problem into a multi-class one."""

    def __init__(self, labels: list[str] | tuple[str, ...]) -> None:
        """Creates the powerset of labels, including the empty label, and different mappings for further use."""
        self._labels = labels
        powerset = [()] + [
            e for i in range(len(labels)) for e in list(combinations(labels, i + 1))
        ]
        self.n_labels = len(powerset)
        self.map = {tuple(sorted(labels)): i for i, labels in enumerate(powerset)}
        """raw label to encoded index value"""
        self.rev_map = {i: labels for labels, i in self.map.items()}

    @property
    def labels(self) -> tuple[tuple[str, ...], ...]:
        """return all created label combinations obtained through powerset transformation.

        Returns:
            tuple[tuple[str, ...], ...]: all combinations of labels, the tuple is of length: `len(PowersetMultiLabelEncoder)`
        """
        return tuple(self.map.keys())

    @property
    def base_labels(self) -> tuple[str, ...]:
        """Return base labels that are encoded"""
        return tuple(self._labels)

    def transform(self, labels: Iterable[str] | str) -> int:
        """Transforms a label or a set of labels into its corresponding integer value representing the position of the label in the `y` one-hot vector used during training.

        Args:
            labels (Iterable[str] | str): labels to transform. `labels` can be a set of labels since the Powerset transforms combinations of labels into a single value.

        Returns:
            int: integer value corresponding to the input labels.
        """
        labels = (labels,) if isinstance(labels, str) else tuple(sorted(labels))
        return self.map[labels]

    def inv_transform(self, i: int) -> tuple[str, ...]:
        """takes an encoded label (integer) and returns the original labels as a tuple of strings.

        Args:
            i (int): The encoded value label, obtained through `self.transform()` or other means.

        Raises:
            ValueError: Raises if the input value is not comprised between 0 <= i < n_labels

        Returns:
            tuple[str, ...]: _description_
        """
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
        return np.eye(self.n_labels, dtype=np.uint8)[self.transform(labels)]

    def i_to_one_hot(self, i: int) -> np.ndarray:
        """takes the encoded values of the encoder and returns the corresponding one-hot vector."""
        labels = self.rev_map[i]
        return self.one_hot(labels)

    def __len__(self) -> int:
        """Returns the size of the generated powerset classes.

        Returns:
            int: _description_
        """
        return len(self.map)

    def __contains__(self, label: str | tuple[str, ...] | list[str]) -> bool:
        if isinstance(label, str):
            label = (label,)
        return tuple(sorted(label)) in self.labels
