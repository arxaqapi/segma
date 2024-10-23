import torch

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
