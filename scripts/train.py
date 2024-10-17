import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from segma.dataloader import Config, SegmentationDataLoader
from segma.models.miniseg import Minisinc
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    l_encoder = PowersetMultiLabelEncoder(
        ["male", "female", "key_child", "other_child"]
    )

    model = Minisinc(l_encoder)
    # model = Miniseg(l_encoder)
    dm = SegmentationDataLoader(l_encoder, config=Config(model.conv_settings))

    trainer = pl.Trainer(
        max_epochs=50,
        # logger=CSVLogger("logs", "miniseg_debug")
        logger=WandbLogger(project="Segma debug", name="train_log"),
    )

    trainer.fit(model, datamodule=dm)
