from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union, get_type_hints

import yaml


@dataclass
class WandbConfig:
    log: bool
    project: str
    name: str


@dataclass
class DataConfig:
    dataset_path: str
    classes: list[str]


@dataclass
class AudioConfig:
    chunk_duration_s: float
    sample_rate: int
    strict_frames: bool


@dataclass
class DataloaderConfig:
    num_workers: int


@dataclass
class SchedulerConfig:
    patience: int


@dataclass
class TrainConfig:
    model: str
    lr: float
    batch_size: int
    max_epochs: int
    validation_metric: str
    profiler: str | None

    dataloader: DataloaderConfig

    scheduler: SchedulerConfig


@dataclass
class Config:
    wandb: WandbConfig
    data: DataConfig
    audio_config: AudioConfig
    train: TrainConfig


T = TypeVar("T")


def load_config(config_path: str | Path, config_class: Type[T] = Config) -> T:
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

    # Load YAML file
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")

    # Create main Config instance
    return create_dataclass_instance(config_class, config_data)
