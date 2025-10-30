import torch
import yaml

from segma.config.base import Config, load_config
from segma.models import Models
from segma.utils.encoders import MultiLabelEncoder


def test_setup_gen_conf():
    _cfg: Config = load_config(
        config_path="src/segma/config/default.yml",
    )
    for model_name, _model_c in Models.items():
        _cfg.model.name = model_name
        _m_dict = _cfg.as_dict()
        _m_dict["model"].pop("config")

        with open(f"tests/sample/temp_config_{model_name}.yml", "w") as f:
            yaml.safe_dump(_m_dict, f)


def test_models():
    labels = ("MAL", "FEM", "KCHI", "OCH")

    for model_name, model_c in Models.items():
        if "hydra" not in model_name:
            raise ValueError("Only `MultiLabelEncoder` is supported")
        label_encoder = MultiLabelEncoder(labels)

        cfg: Config = load_config(
            config_path=f"tests/sample/temp_config_{model_name}.yml",
        )
        model = model_c(label_encoder, cfg)
        assert model is not None


def test_Whisper_based_forward():
    labels = ("aa", "bb", "cc")

    x_t = torch.ones((2, 80, 3000))

    for model_name, model_c in Models.items():
        if (
            "whisper" in model_name or "hydra" in model_name
        ) and "wavlm" not in model_name:
            if "hydra" not in model_name:
                raise ValueError("Only `MultiLabelEncoder` is supported")
            label_encoder = MultiLabelEncoder(labels)

            cfg: Config = load_config(
                config_path=f"tests/sample/temp_config_{model_name}.yml",
            )
            _out = model_c(label_encoder, cfg)(x_t)


def test_WavLM_based_forward():
    labels = ("aa", "bb", "cc")

    # Takes in raw audio
    x_t = torch.ones((2, 32_000))

    for model_name, model_c in Models.items():
        if "wavlm" in model_name:
            if "hydra" not in model_name:
                raise ValueError("Only `MultiLabelEncoder` is supported")
            label_encoder = MultiLabelEncoder(labels)

            cfg: Config = load_config(
                config_path=f"tests/sample/temp_config_{model_name}.yml",
            )
            _out = model_c(label_encoder, cfg)(x_t)


def test_cleanup():
    # NOTE - cleanup
    from pathlib import Path

    for model_name, model_c in Models.items():
        Path(f"tests/sample/temp_config_{model_name}.yml").unlink(missing_ok=True)
