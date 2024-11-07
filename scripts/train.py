import argparse
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from segma.dataloader import Config, SegmentationDataLoader
from segma.models import Models, Whisperidou, WhisperiMax
from segma.utils.encoders import PowersetMultiLabelEncoder


def get_metric(metric: str) -> tuple[str, str]:
    if args.metric == "loss":
        return "min", "val/loss"
    elif args.metric == "auroc":
        return "max", "val/auroc"
    elif args.metric == "fscore":
        return "max", "val/f1_score"
    else:
        raise ValueError(
            f"metric '{metric}' is not supported, please use 'loss', 'auroc' or 'fscore'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb", action="store_true", help="Use Wandb or not (default is set to no)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss", "auroc", "fscore"],
        default="auroc",
        help="Evaluation metric to use (loss, auroc, fscore)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["whisperidou", "whisperimax"],  # , "pyannet"],
        default="whisperimax",
        help="Model to use (whisperidou, whisperimax)",  # , fscore)",
    )
    parser.add_argument(
        "--tags",
        type=list[str],
        default=[],
        help="Tags to be added to the wandb logging instance.",
    )

    args = parser.parse_args()

    chkp_path = Path("models")
    if not chkp_path.exists():
        chkp_path.mkdir()

    labels = ("KCHI", "OCH", "FEM", "MAL", "SPEECH")
    l_encoder = PowersetMultiLabelEncoder(labels)

    model: Whisperidou | WhisperiMax = Models[args.model](l_encoder)

    mode, monitor = get_metric(args.metric)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=1e-3)
        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(optim, mode=mode, patience=3),
            "monitor": monitor,
        }

    model.configure_optimizers = MethodType(configure_optimizers, model)

    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initializing ...",
        flush=True,
    )
    dm = SegmentationDataLoader(
        l_encoder,
        config=Config(model.conv_settings, labels),
        audio_preparation_hook=model.audio_preparation_hook,
    )
    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initialized",
        flush=True,
    )

    reference_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    save_path = Path("models")

    if not args.wandb:
        print("[log] - use CSVLogger")

        save_path = save_path / f"{reference_time}"
        save_path.mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(save_dir=save_path)
    else:
        print("[log] - use WandbLogger")
        from lightning.pytorch.loggers import WandbLogger

        try:
            logger = WandbLogger(
                project="Segma debug",
                name="oberon_train",
                log_model="all",
                tags=args.tags,
            )
            save_path = save_path / f"{reference_time}--{logger.experiment.id}"
        except Exception as _:
            import wandb

            wandb.init(mode="disabled")
            logger = WandbLogger(
                project="Segma debug",
                name="oberon_train",
                log_model="all",
                tags=args.tags,
            )
            save_path = save_path / f"{reference_time}--{logger.experiment.id}"
        save_path.mkdir(parents=True, exist_ok=True)

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
        devices="gpu",
        max_epochs=60,
        logger=logger,
        callbacks=[model_checkpoint, early_stopping, LearningRateMonitor()],
    )

    model = torch.compile(model)

    print(f"[log @ {datetime.now().strftime('%Y%m%d_%H%M')}] - started training")
    trainer.fit(model, datamodule=dm)

    # NOTE - symlink to best model
    (chkp_path / "best.ckpt").symlink_to(
        Path(model_checkpoint.best_model_path).absolute()
    )

    print(f"[log] - best model score: {model_checkpoint.best_model_score}")
    print(f"[log] - best model path: {model_checkpoint.best_model_path}")
