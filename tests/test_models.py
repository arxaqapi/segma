import torch

from segma.config.base import Config, load_config
from segma.models import PyanNet, Whisperidou, WhisperiMax
from segma.utils.encoders import PowersetMultiLabelEncoder


def test_Whisperidou_init():
    labels = ("aa", "bb", "cc")

    cfg: Config = load_config(
        config_path="tests/sample/test_config_whisperidou.yml",
    )

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = Whisperidou(label_encoder, cfg)

    assert model.w_encoder is not None


def test_WhisperiMax_init():
    labels = ("aa", "bb", "cc")

    cfg: Config = load_config(
        config_path="tests/sample/test_config_whisperimax.yml",
    )

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = WhisperiMax(label_encoder, cfg)

    assert model.w_encoder is not None


def test_WhisperiMax_forward():
    labels = ("aa", "bb", "cc")

    cfg: Config = load_config(
        config_path="tests/sample/test_config_whisperimax.yml",
    )

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = WhisperiMax(label_encoder, cfg)

    x = torch.ones((1, 80, 3000))

    out = model(x)

    assert out.shape == (1, 99, 8)


def test_PyanNet_init():
    labels = ("aa", "bb", "cc")

    cfg: Config = load_config(
        config_path="tests/sample/test_config_pyannet.yml",
    )

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = PyanNet(label_encoder, cfg)

    assert model is not None
    assert len(model.linear) == len(model.hparams.linear)


def test_PyanNet_forward():
    labels = ("aa", "bb", "cc")

    cfg: Config = load_config(
        config_path="tests/sample/test_config_pyannet.yml",
    )

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = PyanNet(label_encoder, cfg)

    x = torch.ones((1, 32_000))
    out = model(x)

    # 115
    # print(model.conv_settings.n_windows())

    assert out.shape == (1, model.conv_settings.n_windows(), len(label_encoder.labels))
    assert (1, 115, 8) == (
        1,
        model.conv_settings.n_windows(),
        len(label_encoder.labels),
    )
    assert (1, 118, 8) == (
        1,
        model.conv_settings.n_windows(strict=False),
        len(label_encoder.labels),
    )
