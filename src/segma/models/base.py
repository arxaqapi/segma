from dataclasses import dataclass
from functools import reduce
from math import floor

import lightning as pl
import matplotlib.pyplot as plt
import torch
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

    def n_windows(self, chunk_duration_f: int = 32_000, strict: bool = True) -> int:
        """compute the total number of covered windows for a given audio duration

        Args:
            chunk_duration_f (int, optional): duration of the reference audio. Defaults to 32_000.
            strict (bool, optional): if strict, the last window is fully contained, otherwise allow for overshooting. Defaults to True.

        Returns:
            int: _description_
        """
        # add a correction term (1) to the receptive field step computation (in the case a kernel has an even size).
        correct = reduce(lambda b, e: b or (e % 2 == 0), self.kernels, False)

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

        # NOTE - check shape sizes
        if y_target.shape != y_pred.shape:
            raise ValueError(
                f"y_target and y_predict shapes do not match, got shapes: {y_target.shape=} {y_pred.shape=}"
            )

        loss = torch.nn.functional.cross_entropy(input=y_pred, target=y_target)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # NOTE - split loss for the first n elements
        n_single = len(
            list(filter(lambda e: e < 2, map(len, self.label_encoder.labels)))
        )

        # create a weight vector of size len(self.label_encoder) and put the first n elements to 1.
        weights = torch.zeros(size=(len(self.label_encoder),)).to(self.device)
        weights[:n_single] = 1.0
        with torch.no_grad():
            partial_loss = torch.nn.functional.cross_entropy(
                input=y_pred, target=y_target, weight=weights
            )
        self.log(
            "val/partial_loss",
            partial_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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
