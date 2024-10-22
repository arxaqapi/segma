import argparse
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import lightning as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from segma.dataloader import Config, SegmentationDataLoader
from segma.models import Minisinc, Whisperidou
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", default=False)
    args = parser.parse_args()

    chkp_path = Path("models")
    if not chkp_path.exists():
        chkp_path.mkdir()

    labels = ("KCHI", "OCH", "FEM", "MAL", "SPEECH")
    l_encoder = PowersetMultiLabelEncoder(labels)

    # model = Minisinc(l_encoder)
    # model = Miniseg(l_encoder)
    model = Whisperidou(l_encoder)
    mode = "min"
    monitor = "val/loss"

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

    if args.log:
        logger = WandbLogger(
            project="Segma debug",
            name="oberon_train",
            log_model="all",
            tags=[],
        )

    reference_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    chkp_path = (
        Path("models")
        / (
            f"{reference_time}--{logger.experiment.id}"
            if args.log
            else f"{reference_time}"
        )
        / "checkpoints"
    )
    chkp_path.mkdir(parents=True, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
        dirpath=chkp_path,
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
        max_epochs=60,
        logger=None if not args.log else logger,
        callbacks=[model_checkpoint, early_stopping, LearningRateMonitor()],
    )

    print(f"[log @ {datetime.now().strftime('%Y%m%d_%H%M')}] - started training")
    trainer.fit(model, datamodule=dm)

    # NOTE - symlink to best model
    (chkp_path / "best.ckpt").symlink_to(
        Path(model_checkpoint.best_model_path).absolute()
    )

    print(f"[log] - best model score: {model_checkpoint.best_model_score}")
    print(f"[log] - best model path: {model_checkpoint.best_model_path}")
