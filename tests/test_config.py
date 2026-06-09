from segma.config import Config, load_config


def test_load():
    conf: Config = load_config("tests/sample/test_config.toml")

    assert conf.logging.id is None
    assert conf.model.hubert_checkpoint is None
    assert conf.train.seed is None


def test_load_full():
    conf: Config = load_config("tests/sample/test_config_full.toml")

    assert conf.logging.id == "a6bf87"
    assert conf.model.hubert_checkpoint == "model/best.ckpt"
    assert conf.train.seed == 0
