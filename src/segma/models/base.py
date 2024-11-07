from dataclasses import dataclass
from math import floor

import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from torchmetrics.functional.classification import (
    # auroc,
    # multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
    multiclass_roc,
)

from segma.utils.encoders import LabelEncoder, PowersetMultiLabelEncoder
from segma.utils.receptive_fields import rf_center_i, rf_size


@dataclass
class ConvolutionSettings:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]

    def n_windows(
        self, chunk_duration_f: int = 32_000, strict: bool = True, correct: bool = True
    ) -> int:
        """compute the total number of covered windows for a given audio duration

        Args:
            chunk_duration_f (int, optional): duration of the reference audio. Defaults to 32_000.
            strict (bool, optional): if strict, the last window is fully contained, otherwise allow for overshooting. Defaults to True.
            correct (bool, optional): add a correction term (1) to the recetpive field step computation (in the case a kernel has size even). Defaults to True.

        Returns:
            int: _description_
        """
        # Should be 320 (f) with duration 2 secs and whisper model
        # Should be 270 (f) with duration 2 secs and sinc model
        rf_step = rf_step = int(
            rf_center_i(
                5,
                self.kernels,
                self.strides,
                self.paddings,
            )
            - rf_center_i(
                4,
                self.kernels,
                self.strides,
                self.paddings,
            )
            + (1 if correct else 0)
        )

        return (
            (chunk_duration_f // rf_step)
            if not strict
            else floor(
                (chunk_duration_f - rf_size(self.kernels, self.strides)) / rf_step
            )
            + 1
        )


class BaseSegmentationModel(pl.LightningModule):
    def __init__(self, label_encoder: LabelEncoder) -> None:
        super().__init__()
        if not isinstance(label_encoder, PowersetMultiLabelEncoder):
            raise ValueError(
                "Only PowersetMultiLabelEncoder is accepted at the moment."
            )
        self.label_encoder = label_encoder

    def audio_preparation_hook(self, audio_t):
        """should be overwritten in the child class,
        if audio processing is necessary before passing through the model"""
        return audio_t

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

        average_f_score = multiclass_f1_score(
            preds=y_pred.argmax(-1),
            target=y_target.argmax(-1),
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
        f1_score_p_class = average_f_score = multiclass_f1_score(
            preds=y_pred.argmax(-1),
            target=y_target.argmax(-1),
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

        # ROC curves plotting
        fpr_s, tpr_s, _ = multiclass_roc(
            preds=y_pred,
            target=y_target.argmax(-1),
            num_classes=len(self.label_encoder),
        )

        roc_fig = plt.figure(figsize=(10, 5))
        roc_ax = roc_fig.add_subplot()
        for fpr, tpr, label in zip(fpr_s, tpr_s, self.label_encoder.labels):
            labels_str = " & ".join([e[:3] for e in label]) if label != () else "None"
            roc_ax.plot(
                fpr.cpu(),
                tpr.cpu(),
                label=f"{labels_str} - AUC={{todo}}",
            )
        roc_ax.plot([0, 1], [0, 1], "k--", label="Random classifier: AUC=0.5")
        roc_ax.set_xlabel("False Positive Rate (Sensitivity )")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title(f"ROC Curve at epoch n° {self.current_epoch}")

        roc_ax.legend(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            edgecolor="None",
        )
        roc_fig.tight_layout()

        try:
            self.logger.experiment.log({"ROC_curves": wandb.Image(roc_fig)})
        except Exception as _:
            pass
        plt.close(roc_fig)

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

    def forward(self, x) -> torch.Tensor:
        x = x[:, None, :]
        x = self.head(x)[..., :-1]
        x = self.net(x)
        x = x.transpose(2, 1)
        logits = self.classifier(x)
        return torch.nn.functional.softmax(logits, dim=-1)
