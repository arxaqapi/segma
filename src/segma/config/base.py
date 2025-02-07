from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Type, TypeVar, Union, get_type_hints

import yaml


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


T = TypeVar("T")


def load_config(
    config_path: str | Path,
    model_config_path: str | Path | None = None,
    config_class: Type[T] = Config,
) -> T:
    """
    Load configuration from a YAML file and create nested dataclass instances.

    Args:
        config_path: Path to the YAML configuration file
        config_class: Top-level configuration class (defaults to Config)

    Returns:
        Instance of config_class with all nested dataclasses initialized

    Raises:
        ValueError: If required configuration fields are missing
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """

    def create_dataclass_instance(cls: Type[Any], data: dict[str, Any]) -> Any:
        """
        Recursively create nested dataclass instances from dictionary data.
        """
        if not is_dataclass(cls):
            return data

        field_types = get_type_hints(cls)
        kwargs = {}

        for field in fields(cls):
            field_name = field.name
            if field_name not in data:
                raise ValueError(f"Missing required configuration field: {field_name}")

            field_value = data[field_name]
            field_type = field_types[field_name]

            # Handle Optional/Union types (like str | None)
            if hasattr(field_type, "__origin__"):
                if field_type.__origin__ is Union:
                    if field_value is None:
                        kwargs[field_name] = None
                        continue
                    field_type = field_type.__args__[0]  # Take first type argument

            # If the field is itself a dataclass, recursively create instance
            if is_dataclass(field_type):
                kwargs[field_name] = create_dataclass_instance(field_type, field_value)
            else:
                kwargs[field_name] = field_value

        return cls(**kwargs)

    # Convert string path to Path object
    config_path = Path(config_path)
    shoehorn = False
    # Load YAML file
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        # NOTE - dummy patch to avoid adding `config: null` in `train.model`
        if "config" not in config_data["model"]:
            shoehorn = True
            config_data["model"]["config"] = None
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Uncaught error: {e}")

    # Create main Config instance
    config: Config = create_dataclass_instance(config_class, config_data)

    model_config_class = {
        "pyannet": PyanNetConfig,
        "pyannet_slim": PyanNetSlimConfig,
        "whisperidou": WhisperidouConfig,
        "whisperimax": WhisperimaxConfig,
        "surgical_whisper": SurgicalWhisperConfig,
        "hydra_whisper": HydraWhisperConfig,
        "surgical_hydra": SurgicalHydraConfig,
    }
    # NOTE Manually shoehorn ModelConfig as config.model.config
    if shoehorn:
        if model_config_path is None:
            model_config_path = f"src/segma/config/{config.model.name}.yml"

        with open(model_config_path, "r") as f:
            model_config_data = yaml.safe_load(f)

        config.model.config = create_dataclass_instance(
            model_config_class[config.model.name], model_config_data
        )
    # FIXME - why ?
    if isinstance(config.model.config, dict):
        config.model.config = create_dataclass_instance(
            model_config_class[config.model.name], config.model.config
        )
    return config
