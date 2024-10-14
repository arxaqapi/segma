import numpy as np
import pytest

from segma.utils.encoders import MultiLabelEncoder


def test_m_label_encoder_init():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)

    for label in labels:
        assert label in enc.map.keys()
        assert enc.map[label] < len(labels)


def test_m_label_encoder_labels():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)
    assert enc.labels == labels


def test_m_label_encoder_transform():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)

    assert enc.transform("A") == 0
    assert enc.transform("B") == 1
    assert enc.transform("C") == 2


def test_m_label_encoder_one_hot():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)

    assert list(enc.one_hot("A")) == [1, 0, 0]
    assert list(enc.one_hot("B")) == [0, 1, 0]
    assert list(enc.one_hot("C")) == [0, 0, 1]

    assert list(enc.one_hot(["A", "B"])) == [1, 1, 0]
    assert list(enc.one_hot(["B", "C"])) == [0, 1, 1]
    assert list(enc.one_hot(["A", "A", "C"])) == [1, 0, 1]


def test_m_label_encoder_inv_one_hot():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)

    assert "A" not in enc.inv_one_hot(np.array([0, 1, 0]))
    assert "B" in enc.inv_one_hot(np.array([0, 1, 0]))
    assert "C" not in enc.inv_one_hot(np.array([0, 1, 0]))

    with pytest.raises(NotImplementedError):
        enc.inv_one_hot(np.zeros((2, len(labels))))

    with pytest.raises(ValueError):
        enc.inv_one_hot(np.zeros(len(labels) - 1))


def test_m_label_encoder_length():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)
    assert len(enc) == len(labels)


def test_m_label_encoder_contains():
    labels = ("A", "B", "C")
    enc = MultiLabelEncoder(labels)
    for label in labels:
        assert label in enc
