import torch
import yaml

from segma.config.base import Config, load_config
from segma.models import Models
from segma.utils.encoders import MultiLabelEncoder, PowersetMultiLabelEncoder


def test_models():
    # NOTE - gen config files
    _cfg: Config = load_config(
        config_path="src/segma/config/default.yml",
    )
    for model_name, model_c in Models.items():
        _cfg.model.name = model_name
        _m_dict = _cfg.as_dict()
        _m_dict["model"].pop("config")

        with open(f"tests/sample/temp_config_{model_name}.yml", "w") as f:
            yaml.safe_dump(_m_dict, f)

    # GEN CONFIGs
    labels = ("MAL", "FEM", "KCHI", "OCH")
    # x_t = torch.ones(32_000)

    for model_name, model_c in Models.items():
        label_encoder = (
            MultiLabelEncoder(labels)
            if "hydra" in model_name
            else PowersetMultiLabelEncoder(labels)
        )
        cfg: Config = load_config(
            config_path=f"tests/sample/temp_config_{model_name}.yml",
        )
        model = model_c(label_encoder, cfg)
        assert model is not None

    # NOTE - cleanup
    # for model_name, model_c in Models.items():
    #     Path(f"tests/sample/temp_config_{model_name}.yml").unlink(missing_ok=True)


def test_Whisper_based_forward():
    labels = ("aa", "bb", "cc")

    x_t = torch.ones((1, 80, 3000))

    for model_name, model_c in Models.items():
        if "whisper" in model_name or "hydra" in model_name:
            label_encoder = (
                MultiLabelEncoder(labels)
                if "hydra" in model_name
                else PowersetMultiLabelEncoder(labels)
            )
            cfg: Config = load_config(
                config_path=f"tests/sample/temp_config_{model_name}.yml",
            )
            _out = model_c(label_encoder, cfg)(x_t)
