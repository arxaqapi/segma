import numpy as np
import pytest

from segma.utils.encoders import MultiLabelEncoder


def test_m_label_encoder_init():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)

    for label in labels:
        assert label in enc.map.keys()
        assert enc.map[label] < len(labels)


def test_m_label_encoder_labels():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)
    assert enc.labels == labels


def test_m_label_encoder_transform():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)

    assert enc.transform("a") == 0
    assert enc.transform("b") == 1
    assert enc.transform("c") == 2


def test_m_label_encoder_one_hot():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)

    assert enc.one_hot("a").tolist() == [1, 0, 0]
    assert enc.one_hot("b").tolist() == [0, 1, 0]
    assert enc.one_hot("c").tolist() == [0, 0, 1]

    assert enc.one_hot(["a", "b"]).tolist() == [1, 1, 0]
    assert enc.one_hot(["b", "c"]).tolist() == [0, 1, 1]
    assert enc.one_hot(["a", "a", "c"]).tolist() == [1, 0, 1]


def test_m_label_encoder_inv_one_hot():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)

    assert "a" not in enc.inv_one_hot(np.array([0, 1, 0]))
    assert "b" in enc.inv_one_hot(np.array([0, 1, 0]))
    assert "c" not in enc.inv_one_hot(np.array([0, 1, 0]))

    with pytest.raises(NotImplementedError):
        enc.inv_one_hot(np.zeros((2, len(labels))))

    with pytest.raises(ValueError):
        enc.inv_one_hot(np.zeros(len(labels) - 1))


def test_m_label_encoder_length():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)
    assert len(enc) == len(labels)


def test_m_label_encoder_contains():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels)
    for label in labels:
        assert label in enc
