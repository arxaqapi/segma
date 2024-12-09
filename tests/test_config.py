import pytest

from segma.config import load_config


def test_load_config():
    cfg = load_config("tests/sample/test_config.yml")

    assert cfg is not None

    assert cfg.wandb.log
    assert cfg.wandb.project == "debug"
    assert cfg.wandb.name == "train"

    assert cfg.data.dataset_path == "data/baby_train"
    assert cfg.data.classes == ["KCHI", "OCH", "MAL", "FEM"]

    assert cfg.audio_config.chunk_duration_s == 2.0
    assert cfg.audio_config.sample_rate == 16_000
    assert cfg.audio_config.strict_frames

    assert cfg.train.model == "whisperidou"
    assert cfg.train.lr == 1e-3
    assert cfg.train.max_epochs == 100
    assert cfg.train.validation_metric == "auroc"
    assert cfg.train.profiler is None
    assert cfg.train.dataloader.num_workers == 8
    assert cfg.train.scheduler.patience == 3


def test_load_config_missing_vals():
    with pytest.raises(ValueError):
        load_config("tests/sample/test_broken_config.yml")


def test_Config_as_dict():
    import yaml

    cfg = load_config("tests/sample/test_config.yml")
    d = cfg.as_dict()

    with open("tests/sample/test_config.yml", "r") as f:
        initial_d = yaml.safe_load(f)

    assert d is not None
    assert initial_d == d
