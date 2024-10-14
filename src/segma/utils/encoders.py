import numpy as np


class MultiLabelEncoder:
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

    def one_hot(self, label: str | list[str] | tuple[str, ...]) -> np.ndarray:
        """Returns the one-hot representation of the label of list of labels given as input.

        Args:
            label (str | list[str]): the label.s to transform into the respective one-hot representation.

        Raises:
            ValueError: Fails if the label is not of type string or list of strings.

        Returns:
            np.ndarray: numpy array of size n.
        """
        _n = len(self.map)
        if isinstance(label, list) or isinstance(label, tuple):
            return (
                np.eye(_n, dtype=np.uint8)[[self.transform(lab) for lab in label]]
                .sum(axis=0)
                .clip(0, 1)
            )
        elif isinstance(label, str):
            return np.eye(_n, dtype=np.uint8)[self.transform(label)]
        else:
            raise ValueError(
                f"input argument {label=} has incorrect type '{type(label)}', used str or list[str] as argument."
            )

    def __call__(self, label: str) -> int:
        return self.transform(label)

    def __len__(self) -> int:
        return len(self.map)

    def __contains__(self, x) -> bool:
        return x in self.labels
