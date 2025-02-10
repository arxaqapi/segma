from pathlib import Path

import pytest
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


def test_load_config():
    cfg = load_config(
        config_path="tests/sample/temp_config_whisperidou.yml",
    )

    assert cfg

    assert cfg.wandb.offline in [True, False]
    assert cfg.wandb.project == "Segma debug"
    assert cfg.wandb.name == "train"

    assert cfg.data.dataset_path == "data/baby_train"
    assert cfg.data.classes == ["KCHI", "OCH", "MAL", "FEM"]
    assert cfg.data.dataset_multiplier == 1.0

    assert cfg.audio.chunk_duration_s == 2.0
    assert cfg.audio.sample_rate == 16_000
    assert not cfg.audio.strict_frames

    assert cfg.model.name == "whisperidou"
    assert cfg.model.config.encoder == "whisper_tiny_encoder"
    assert cfg.model.config.linear == [256]
    assert cfg.model.config.classifier == 256

    assert cfg.train.lr == 1e-3
    assert cfg.train.max_epochs == 100
    assert cfg.train.validation_metric == "auroc"
    assert cfg.train.profiler is None
    assert cfg.train.dataloader.num_workers == 8
    assert cfg.train.scheduler.patience == 3


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config(
            config_path="tests/sample/does_not_exist.yml",
        )


def test_Config_as_dict():
    import yaml

    cfg = load_config(
        config_path="tests/sample/temp_config_whisperidou.yml",
    )
    d = cfg.as_dict()
    del d["model"]["config"]

    with open("tests/sample/temp_config_whisperidou.yml", "r") as f:
        initial_d = yaml.safe_load(f)

    assert d is not None
    assert initial_d == d


def test_Config_save_load():
    """Ensures that Config.save and Config.load works and that a complete config
    is correctly loaded.
    """
    cfg = load_config(
        config_path="tests/sample/temp_config_whisperidou.yml",
    )
    cfg.model.config.encoder = "whisper_large_encoder"
    temp_p = Path("tests/_temp.yml")
    cfg.save(temp_p)

    cfg_saved = load_config("tests/_temp.yml")

    assert cfg.as_dict() == cfg_saved.as_dict()

    temp_p.unlink()


def test_Config_SurgicalWhisperConfig():
    cfg = load_config(
        config_path="tests/sample/temp_config_surgical_whisper.yml",
    )

    assert cfg.model.name == "surgical_whisper"

    assert hasattr(cfg.model.config, "encoder")
    assert hasattr(cfg.model.config, "encoder_layers")
    assert hasattr(cfg.model.config, "reduction")
    assert hasattr(cfg.model.config, "linear")
    assert hasattr(cfg.model.config, "classifier")


def test_load_config_extra_args():
    cfg = load_config(
        config_path="tests/sample/temp_config_surgical_whisper.yml",
        cli_extra_args=[
            "wandb.offline=true",
            "audio.chunk_duration_s=6.0",
            "data.dataset_multiplier=0.2",
            "model.config.encoder_layers=[1,3]",
            "model.config.reduction=average",
            "train.extra_val_metrics=[loss]",
        ],
    )

    assert cfg.wandb.offline
    assert cfg.audio.chunk_duration_s == 6.0
    assert cfg.data.dataset_multiplier == 0.2
    assert cfg.model.config.encoder_layers == [1, 3]
    assert cfg.model.config.reduction == "average"
    assert cfg.train.extra_val_metrics == ["loss"]


def test_cleanup():
    # NOTE - cleanup
    from pathlib import Path

    for model_name, _model_c in Models.items():
        Path(f"tests/sample/temp_config_{model_name}.yml").unlink(missing_ok=True)
