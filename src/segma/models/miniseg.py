from dataclasses import dataclass
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import f1_score, multiclass_auroc

from segma.utils.encoders import LabelEncoder, PowersetMultiLabelEncoder


@dataclass
class ConvolutionSettings:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]


class BaseSegmentationModel(pl.LightningModule):
    def __init__(self, label_encoder: LabelEncoder) -> None:
        super().__init__()
        if not isinstance(label_encoder, PowersetMultiLabelEncoder):
            raise ValueError(
                "Only PowersetMultiLabelEncoder is accepted at the moment."
            )
        self.label_encoder = label_encoder

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred = self.forward(x)

        # reduce first 2 dimensions
        y_target = y_target.view(-1, len(self.label_encoder.labels))
        y_pred = y_pred.view(-1, len(self.label_encoder.labels))

        ############
        # print("===========================")
        # print("> y_targets...")
        # for l in y_target[5:15].tolist():
        #     print([int(e) for e in l])
        # print("> y_predictions...")
        # # for i in y_pred[5:15].argmax(dim=-1):
        # #       print(self.label_encoder.i_to_one_hot(int(i.item())))
        # for i in y_pred[5:15]:
        #     print(i)
        # print("===========================")
        ############

        loss = torch.nn.functional.cross_entropy(input=y_pred, target=y_target)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred = self.forward(x)

        y_target = y_target.view(-1, len(self.label_encoder.labels))
        y_pred = y_pred.view(-1, len(self.label_encoder.labels))

        loss = torch.nn.functional.cross_entropy(input=y_pred, target=y_target)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        average_f_score = f1_score(
            preds=y_pred.argmax(-1),
            target=y_target.argmax(-1),
            task="multiclass",
            num_classes=len(self.label_encoder),
            zero_division=0,
        )
        self.log(
            "val/f1_score",
            average_f_score,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        f1_score_p_class = average_f_score = f1_score(
            preds=y_pred.argmax(-1),
            target=y_target.argmax(-1),
            task="multiclass",
            num_classes=len(self.label_encoder),
            average=None,
            zero_division=0,
        )
        for i, c_f1_score in enumerate(f1_score_p_class):
            c_f1_score = round(c_f1_score.item(), 6)
            label_name = " & ".join(self.label_encoder.inv_transform(i))
            self.log(
                f"val/f1_score/{label_name}",
                c_f1_score,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        avg_auroc = multiclass_auroc(
            preds=y_pred,
            target=y_target.argmax(-1),
            num_classes=len(self.label_encoder),
        )
        self.log(
            "val/auroc",
            avg_auroc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Miniseg(BaseSegmentationModel):
    def __init__(self, label_encoder: LabelEncoder) -> None:
        super().__init__(label_encoder)

        self.head = nn.Conv1d(1, 80, kernel_size=400, stride=160, padding=200)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(80, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Linear(512, len(label_encoder.labels))

        self.conv_settings = ConvolutionSettings(
            kernels=(400, 3, 3), strides=(160, 1, 2), paddings=(200, 1, 1)
        )

    def forward(self, x) -> Any:
        x = x[:, None, :]
        x = self.head(x)[..., :-1]
        x = self.net(x)
        x = x.transpose(2, 1)
        logits = self.classifier(x)
        return torch.nn.functional.softmax(logits, dim=-1)


class Minisinc(BaseSegmentationModel):
    def __init__(self, label_encoder: LabelEncoder) -> None:
        super().__init__(label_encoder)
        # assert isinstance(label_encoder, PowersetMultiLabelEncoder)

        self.net = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=251, stride=10, padding=0),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(80, affine=True),
            nn.LeakyReLU(),
            nn.Conv1d(80, 60, 5, stride=1),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(60, affine=True),
            nn.LeakyReLU(),
            nn.Conv1d(60, 60, 5, stride=1),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(60, affine=True),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(60, len(label_encoder.labels))

        self.conv_settings = ConvolutionSettings(
            kernels=(251, 3, 5, 3, 5, 3), strides=(10, 3, 1, 3, 1, 3), paddings=(0,) * 6
        )

    def forward(self, x):
        x = x[:, None, :]
        x = self.net(x)
        x = x.transpose(2, 1)
        logits = self.classifier(x)
        return torch.nn.functional.softmax(logits, dim=-1)
