import torch

from segma.imports import PyanNet
from segma.models import Whisperidou, WhisperiMax
from segma.utils.encoders import PowersetMultiLabelEncoder


def test_Whisperidou_init():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = Whisperidou(label_encoder)

    assert model.w_encoder is not None


def test_WhisperiMax_init():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = WhisperiMax(label_encoder)

    assert model.w_encoder is not None


def test_WhisperiMax_forward():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = WhisperiMax(label_encoder)

    x = torch.ones((1, 80, 3000))

    out = model(x)

    assert out.shape == (1, 99, 8)


def test_PyanNet_init():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = PyanNet(label_encoder)

    assert model is not None
    assert (
        len(model.linear)
        == model.hparams.linear["num_layers"]
        == model.LINEAR_DEFAULTS["num_layers"]
    )


def test_PyanNet_forward():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = PyanNet(label_encoder)

    x = torch.ones((1, 1, 32_000))
    out = model(x)
    # print(out.shape)

    print(model.conv_settings.n_windows())

    assert out.shape == (1, model.conv_settings.n_windows(), len(label_encoder.labels))
    assert (1, 115, 8) == (
        1,
        model.conv_settings.n_windows(),
        len(label_encoder.labels),
    )
