import torch
import yaml

from segma.config.base import Config, load_config
from segma.models import Models


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
    for model_name, model_c in Models.items():
        cfg: Config = load_config(
            config_path=f"tests/sample/temp_config_{model_name}.yml",
        )
        model = model_c(cfg)
        assert model is not None
        assert model.label_encoder.base_labels == ("KCHI", "OCH", "MAL", "FEM")


# https://docs.pytest.org/en/6.2.x/parametrize.html
# https://medium.com/optuna/overview-of-python-free-threading-v3-13t-support-in-optuna-ad9ab62a11ba
def test_Whisper_based_forward():
    x_t = torch.ones((1, 80, 3000))

    for model_name, model_c in Models.items():
        if "whisper" in model_name or "hydra" in model_name:
            cfg: Config = load_config(
                config_path=f"tests/sample/temp_config_{model_name}.yml",
            )
            model = model_c(cfg)
            model(x_t)


def test_cleanup():
    # NOTE - cleanup
    from pathlib import Path

    for model_name, model_c in Models.items():
        Path(f"tests/sample/temp_config_{model_name}.yml").unlink(missing_ok=True)
