import argparse
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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
    dm = SegmentationDataLoader(
        l_encoder,
        config=Config(model.conv_settings, labels),
        audio_preparation_hook=model.audio_preparation_hook,
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        every_n_epochs=1,
        save_last=True,
        dirpath=chkp_path,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        logger=None
        if not args.log
        else WandbLogger(project="Segma debug", name="train_log"),
        callbacks=[model_checkpoint],
    )

    trainer.fit(model, datamodule=dm)
