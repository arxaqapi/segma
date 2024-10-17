import pytest

from segma.utils.encoders import PowersetMultiLabelEncoder


def test_PowersetMultiLabelEncoder_init():
    labels = ("a", "b", "c")
    enc = PowersetMultiLabelEncoder(labels=labels)

    for label in labels:
        assert (label,) in enc.map.keys()
        assert len(enc.map.keys()) == (1 + 3 + 4)


def test_PowersetMultiLabelEncoder_transform():
    labels = ("aa", "bb", "cc")
    enc = PowersetMultiLabelEncoder(labels=labels)

    assert enc() == enc(())
    assert enc(()) == 0
    assert enc(("aa",)) == 1

    assert enc(("aa", "bb")) == 4
    assert enc(("aa", "cc")) == 5
    assert enc(("aa", "bb", "cc")) == len(enc) - 1


def test_PowersetMultiLabelEncoder_inv_transform():
    labels = ("aa", "bb", "cc")
    enc = PowersetMultiLabelEncoder(labels=labels)

    assert enc.inv_transform(0) == ()
    assert enc.inv_transform(1) == ("aa",)
    assert enc.inv_transform(2) == ("bb",)
    assert enc.inv_transform(4) == ("aa", "bb")

    assert enc.inv_transform(len(enc) - 1) == ("aa", "bb", "cc")

    with pytest.raises(ValueError):
        enc.inv_transform(10)


def test_PowersetMultiLabelEncoder_one_hot():
    labels = ("aa", "bb", "cc")
    enc = PowersetMultiLabelEncoder(labels=labels)

    assert enc.one_hot("aa").tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert enc.one_hot(("aa", "cc")).tolist() == [0, 0, 0, 0, 0, 1, 0, 0]

    assert enc.one_hot(("aa", "bb", "cc")).tolist() == [0, 0, 0, 0, 0, 0, 0, 1]

    # NOTE - order should not be important
    assert (
        enc.one_hot(("aa", "cc", "bb")).tolist()
        == enc.one_hot(("aa", "bb", "cc")).tolist()
    )
    assert enc.one_hot(("bb", "cc")).tolist() == enc.one_hot(("cc", "bb")).tolist()


def test_PowersetMultiLabelEncoder_i_to_one_hot():
    labels = ("aa", "bb", "cc")
    enc = PowersetMultiLabelEncoder(labels=labels)

    assert enc.i_to_one_hot(0).tolist() == enc.one_hot(()).tolist()
    assert enc.i_to_one_hot(1).tolist() == enc.one_hot("aa").tolist()


def test_PowersetMultiLabelEncoder_contains():
    labels = ("aa", "bb", "cc")
    enc = PowersetMultiLabelEncoder(labels=labels)

    assert () in enc
    for label in labels:
        assert label in enc
    assert ("aa", "cc") in enc
    assert ("cc", "aa") in enc
    assert ("cc", "ff") not in enc
