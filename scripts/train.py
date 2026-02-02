import argparse
from pathlib import Path

import lightning as L

# import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from segma.config import load_config
from segma.data import SegmaFileDataset, SegmentationDataLoader
from segma.models import VTC2, MultiLabelModel
from segma.utils import set_seed
from segma.utils.encoders import MultiLabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segma train script argument parser")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config.toml")

    if config.train.seed is not None:
        set_seed(config.train.seed)

    # instantiate model
    ml_encoder = MultiLabelEncoder(config.data.labels)
    model = MultiLabelModel(VTC2(ml_encoder, config.model), config.train)

    # load data
    sfd = SegmaFileDataset.from_config(config)
    sfd.load()

    dm = SegmentationDataLoader(
        dataset=sfd,
        label_encoder=ml_encoder,
        config=config,
        conv_settings=model.model.conv_settings,
    )

    logger = WandbLogger(
        id=config.logging.id,
        name=config.logging.name,
        project=config.logging.project,
        log_model=False if config.logging.offline else "all",
        tags=config.logging.tags,
        offline=config.logging.offline,
    )
    logger.experiment.config.update(config.as_dict())

    (checkpoint_path / "steps").mkdir(parents=True, exist_ok=True)
    periodic_checkpoint = ModelCheckpoint(
        every_n_train_steps=500,
        save_top_k=-1,
        dirpath=checkpoint_path / "steps",
        filename="step={step:8d}",
        auto_insert_metric_name=False,
    )
    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=-1,
        save_last=True,
        mode="min",
        dirpath=checkpoint_path,
        filename="epoch={epoch:03d}-step={step}-val_loss={val/loss:.3f}",
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        min_delta=0.0,
        patience=config.train.scheduler_patience,
        strict=True,
        verbose=False,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epochs,
        logger=logger,
        callbacks=[
            periodic_checkpoint,
            best_checkpoint,
            early_stopping,
            LearningRateMonitor(),
            TQDMProgressBar(1000),
        ],
    )

    # model = torch.compile(model)

    trainer.fit(model, datamodule=dm)

    # NOTE - save best
    (checkpoint_path / "best.ckpt").symlink_to(
        Path(best_checkpoint.best_model_path).absolute()
    )
