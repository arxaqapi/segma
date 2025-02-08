from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import dacite
import yaml
from omegaconf import OmegaConf


@dataclass
class BaseConfig:
    def as_dict(self) -> dict:
        """Returns the dict representation of the current instance of a configuration `Config`.

        Returns:
            dict: dict containing the configuration values loaded from disk, or with subsequent modifications.
        """
        return asdict(self)

    def save(self, file_path: str | Path) -> None:
        """Save current configuration to a YAML file.

        Args:
            file_path (str | Path): Path where to save the YAML file

        Raises:
            IOError: If unable to write to the specified path
        """
        file_path = Path(file_path)

        try:
            with file_path.open("w") as f:
                yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        except IOError as e:
            raise IOError(f"Failed to write configuration to {file_path}: {e}")


@dataclass
class WandbConfig(BaseConfig):
    offline: bool
    project: str
    name: str


@dataclass
class DataConfig(BaseConfig):
    dataset_path: str
    classes: list[str]


@dataclass
class AudioConfig(BaseConfig):
    chunk_duration_s: float
    sample_rate: int
    strict_frames: bool

    @property
    def chunk_duration_f(self) -> int:
        """Returns the amount of frames in a chunk of duration `chunk_duration_s`."""
        # seconds_to_frames
        return int(self.chunk_duration_s * self.sample_rate)


@dataclass
class DataloaderConfig(BaseConfig):
    num_workers: int


@dataclass
class SchedulerConfig(BaseConfig):
    patience: int


@dataclass
class LSTMConfig(BaseConfig):
    hidden_size: int
    num_layers: int
    bidirectional: int
    dropout: int


@dataclass
class SincNetConfig(BaseConfig):
    stride: int


@dataclass
class PyanNetConfig(BaseConfig):
    sincnet: SincNetConfig
    lstm: LSTMConfig
    linear: list[int]
    classifier: int


@dataclass
class PyanNetSlimConfig(BaseConfig):
    sincnet: SincNetConfig
    linear: list[int]
    classifier: int


@dataclass
class WhisperidouConfig(BaseConfig):
    encoder: str
    linear: list[int]
    classifier: int


@dataclass
class WhisperimaxConfig(BaseConfig):
    encoder: str
    lstm: LSTMConfig
    linear: list[int]
    classifier: int


@dataclass
class SurgicalWhisperConfig(BaseConfig):
    encoder: str
    encoder_layers: list[int]
    reduction: Literal["average", "weighted"]
    linear: list[int]
    classifier: int


@dataclass
class HydraWhisperConfig(BaseConfig):
    encoder: str
    lstm: LSTMConfig
    classifier: int


@dataclass
class SurgicalHydraConfig(BaseConfig):
    encoder: str
    encoder_layers: list[int]
    reduction: Literal["average", "weighted"]
    lstm: LSTMConfig
    classifier: int


@dataclass
class ModelConfig(BaseConfig):
    name: str
    # is initialized as None in first pass, then as the correct model class manually (sub-optimal)
    config: (
        None
        | PyanNetConfig
        | PyanNetSlimConfig
        | WhisperidouConfig
        | WhisperimaxConfig
        | SurgicalWhisperConfig
        | HydraWhisperConfig
        | SurgicalHydraConfig
    )


@dataclass
class TrainConfig(BaseConfig):
    lr: float
    batch_size: int
    max_epochs: int
    validation_metric: str
    extra_val_metrics: list[str]
    profiler: str | None

    dataloader: DataloaderConfig

    scheduler: SchedulerConfig


@dataclass
class Config(BaseConfig):
    wandb: WandbConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    train: TrainConfig


def _merge_dict(source, destination):
    """Merge 2 dict.
    - https://stackoverflow.com/a/20666342
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _merge_dict(value, node)
        else:
            destination[key] = value

    return destination


def load_config(config_path: Path, cli_extra_args: list[str]) -> Config:
    config_path = Path(config_path)
    # NOTE - load model config
    with config_path.open("r") as f:
        config_d = yaml.safe_load(f)
    # NOTE - add model config to dict
    model_c_p = Path(f"src/segma/config/{config_d['model']['name']}.yml")
    if not model_c_p.exists():
        raise ValueError(
            f"Model config dict of model {config_d['model']['name']}, could not be loaded"
        )
    with model_c_p.open("r") as f:
        config_d["model"]["config"] = yaml.safe_load(f)
    # NOTE - merge with extra_args_dict
    config_d = OmegaConf.merge(config_d, OmegaConf.from_cli(cli_extra_args))
    config_d = OmegaConf.to_object(config_d)
    # NOTE - attempt to recursively instantiate
    return dacite.from_dict(data_class=Config, data=config_d)
