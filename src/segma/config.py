from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping

import tomlkit

from segma.utils.conversions import seconds_to_frames


# compatible with Wandb
@dataclass(frozen=True)
class LoggingConfig:
    project: str
    name: str
    id: str | None = None
    offline: bool = True
    tags: List[str] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    labels: list[str]
    dataset_multiplier: int | float = 1


@dataclass(frozen=True)
class AudioConfig:
    chunk_duration_s: float = 4.0
    sample_rate: int = 16_000
    strict_frames: bool = False  # False if Whisper

    @property
    def chunk_duration_f(self) -> float:
        """Returns the amount of frames in a chunk of duration `chunk_duration_s`."""
        return seconds_to_frames(self.chunk_duration_s, self.sample_rate)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 10

    dataload_num_workers: int = 16
    scheduler_patience: int = 10
    seed: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    hubert_checkpoint: Path


@dataclass(frozen=True)
class Config:
    logging: LoggingConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    train: TrainConfig

    def save(self, out: Path) -> None:
        # out.write_text(tomlkit.dumps(asdict(self)))
        out.write_text(tomlkit.dumps(self.as_dict()))

    def as_dict(self):
        def _to_builtin(obj):
            """Recursively convert dataclass to dict with only built-in types."""
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    k: _to_builtin(getattr(obj, k)) for k in obj.__dataclass_fields__
                }
            elif isinstance(obj, Mapping):
                return {k: _to_builtin(v) for k, v in obj.items()}
            elif (
                isinstance(obj, (list, tuple))
                or hasattr(obj, "__iter__")
                and not isinstance(obj, (str, bytes))
            ):
                return [_to_builtin(v) for v in obj]
            elif isinstance(obj, Path):
                return Path(obj)
            else:
                return obj

        return _to_builtin(self)


def config_from_mapping(data: Mapping) -> Config:
    return Config(
        logging=LoggingConfig(**data["logging"]),
        data=DataConfig(**data["data"]),
        audio=AudioConfig(**data["audio"]),
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**data["train"]),
    )


def load_config(config_path: Path) -> Config:
    config_path = Path(config_path)
    if not config_path.suffix == ".toml":
        raise ValueError("Config file should be a toml file")
    data = tomlkit.loads(config_path.read_text(encoding="utf-8"))

    return config_from_mapping(data)
