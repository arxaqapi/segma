import argparse
import time
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Literal

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from segma.config import Config, load_config
from segma.data import SegmaFileDataset, SegmentationDataLoader
from segma.models import (
    HydraWavLM,
    HydraWhisper,
    Models,
    PyanNet,
    PyanNetSlim,
    SurgicalHydraWavLM,
    SurgicalWhisper,
    Whisperidou,
    WhisperiMax,
)
from segma.utils import set_seed
from segma.utils.encoders import MultiLabelEncoder, PowersetMultiLabelEncoder


def get_metric(metric: str) -> tuple[Literal["min", "max"], str]:
    match metric:
        case "loss":
            return "min", "val/loss"
        case "f1_score":
            return "max", "val/f1_score"
        case "auroc":
            return "max", "val/auroc"
        case _:
            raise ValueError(
                f"metric '{metric}' is not supported, please use 'loss', 'auroc' or 'f1_score'."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/segma/config/default.yml",
        help="Config file to be loaded and used for the training.",
    )
    parser.add_argument(
        "-mc",
        "--model-config",
        type=str,
        default=None,
        help="Config file to be loaded and used for the training.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to be added to the wandb logging instance.",
    )

    args, extra_args = parser.parse_known_args()
    cfg: Config = load_config(config_path=args.config, cli_extra_args=extra_args)
    if cfg.train.seed:
        set_seed(cfg.train.seed)

    chkp_path = Path(cfg.model.chkp_pth)
    # FIXME - mkdir or not ?
    if not chkp_path.exists():
        chkp_path.mkdir()

    if "hydra" in cfg.model.name:
        l_encoder = MultiLabelEncoder(labels=cfg.data.classes)
    else:
        l_encoder = PowersetMultiLabelEncoder(labels=cfg.data.classes)

    model: (
        Whisperidou
        | WhisperiMax
        | PyanNet
        | PyanNetSlim
        | SurgicalWhisper
        | HydraWhisper
        | HydraWavLM
        | SurgicalHydraWavLM
    ) = Models[cfg.model.name](l_encoder, cfg)

    mode, monitor = get_metric(cfg.train.validation_metric)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=cfg.train.lr)
        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(
                optim, mode=mode, patience=cfg.train.scheduler.patience
            ),
            "monitor": monitor,
        }

    model.configure_optimizers = MethodType(configure_optimizers, model)

    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initializing ...",
        flush=True,
    )

    sfd = SegmaFileDataset.from_config(cfg)
    sfd.load()

    dm = SegmentationDataLoader(
        dataset=sfd,
        label_encoder=l_encoder,
        config=cfg,
        conv_settings=model.conv_settings,
        audio_preparation_hook=model.audio_preparation_hook,
    )
    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initialized",
        flush=True,
    )

    reference_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    save_path = chkp_path / f"{reference_time}"

    print("[log] - use WandbLogger")

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        log_model=False if cfg.wandb.offline else "all",
        tags=args.tags,
        offline=cfg.wandb.offline,
    )
    logger.experiment.config.update(cfg.as_dict())
    save_path = save_path.with_stem(save_path.stem + f"-{logger.experiment.id}")

    save_path.mkdir(parents=True, exist_ok=True)
    cfg.save(save_path / "config.yml")

    chkp_path = save_path / "checkpoints"
    chkp_path.mkdir(parents=True, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=-1,
        # every_n_epochs=1,
        save_last=True,
        dirpath=chkp_path,
        filename="epoch={epoch:02d}-val_loss={val/loss:.3f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        callbacks=[
            model_checkpoint,
            early_stopping,
            LearningRateMonitor(),
            TQDMProgressBar(1000 if "debug" not in cfg.data.dataset_path else 1),
        ],
        # profiler="advanced"
        profiler=cfg.train.profiler,
    )

    # https://pytorch.org/docs/main/torch.compiler_troubleshooting.html#dealing-with-recompilations
    # https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.t130sdb4rshr
    if cfg.model.name in ("hydra_whisper", "HydraWhisper"):
        torch._dynamo.config.accumulated_cache_size_limit = 32
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 32
        model = torch.compile(model)

    print(f"[log @ {datetime.now().strftime('%Y%m%d_%H%M')}] - started training")
    trainer.fit(model, datamodule=dm)

    # NOTE - symlink to best model and to static best model (models/last/best.ckpt)
    (chkp_path / "best.ckpt").symlink_to(
        Path(model_checkpoint.best_model_path).absolute()
    )
    static_p = Path("models/last")
    static_p.mkdir(parents=True, exist_ok=True)
    bm_static_p = static_p / "best.ckpt"
    bm_static_p.unlink(missing_ok=True)
    bm_static_p.symlink_to(Path(model_checkpoint.best_model_path).absolute())

    print(f"[log] - best model score: {model_checkpoint.best_model_score}")
    print(f"[log] - best model path: {model_checkpoint.best_model_path}")
