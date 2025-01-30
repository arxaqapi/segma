import pytest

from segma.utils.encoders import MultiLabelEncoder


def test_MultiLabelEncoder_init():
    labels = ("a", "b", "c")
    enc = MultiLabelEncoder(labels=labels)

    for label in labels:
        assert label in enc.map.keys()
        assert len(enc.map.keys()) == 3


def test_MultiLabelEncoder_transform():
    labels = ("aa", "bb", "cc")
    enc = MultiLabelEncoder(labels=labels)

    assert enc("aa") == 0
    assert enc("bb") == 1
    assert enc("cc") == 2


def test_MultiLabelEncoder_inv_transform():
    labels = ("aa", "bb", "cc")
    enc = MultiLabelEncoder(labels=labels)

    assert enc.inv_transform(0) == "aa"
    assert enc.inv_transform(1) == "bb"
    assert enc.inv_transform(2) == "cc"

    assert enc.inv_transform(len(enc) - 1) == "cc"

    with pytest.raises(ValueError):
        enc.inv_transform(10)
        enc.inv_transform(-10)


def test_MultiLabelEncoder_one_hot():
    labels = ("aa", "bb", "cc")
    enc = MultiLabelEncoder(labels=labels)

    assert enc.one_hot(()).tolist() == [0, 0, 0]
    assert enc.one_hot("aa").tolist() == [1, 0, 0]
    assert enc.one_hot(("aa", "cc")).tolist() == [1, 0, 1]

    assert enc.one_hot(("aa", "bb", "cc")).tolist() == [1, 1, 1]

    # NOTE - order should not be important
    assert (
        enc.one_hot(("aa", "cc", "bb")).tolist()
        == enc.one_hot(("aa", "bb", "cc")).tolist()
    )
    assert enc.one_hot(("bb", "cc")).tolist() == enc.one_hot(("cc", "bb")).tolist()


def test_PowersetMultiLabelEncoder_i_to_one_hot():
    labels = ("aa", "bb", "cc")
    enc = MultiLabelEncoder(labels=labels)

    assert enc.i_to_one_hot(0).tolist() == enc.one_hot("aa").tolist()
    assert enc.i_to_one_hot(1).tolist() == enc.one_hot("bb").tolist()
    assert enc.i_to_one_hot(2).tolist() == enc.one_hot("cc").tolist()


def test_MultiLabelEncoder_contains():
    labels = ("aa", "bb", "cc")
    enc = MultiLabelEncoder(labels=labels)

    for label in labels:
        assert label in enc

    assert "ayh" not in enc
