from dataclasses import dataclass
from functools import reduce
from math import floor

import lightning as pl
import matplotlib.pyplot as plt
import torch
import wandb
from torchmetrics.functional.classification import (
    multiclass_auroc,
    multiclass_f1_score,
    multiclass_roc,
)

from segma.config.base import Config
from segma.utils.encoders import LabelEncoder
from segma.utils.receptive_fields import rf_center_i, rf_size


@dataclass
class ConvolutionSettings:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]

    def n_windows(self, chunk_duration_f: int, strict: bool = True) -> int:
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
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool
    ) -> None:
        super().__init__()
        self.label_encoder = label_encoder
        self.config = config
        self.weights = (
            torch.tensor(
                [0.4] + [1] * (len(self.label_encoder.labels) - 1),
                device=torch.device(
                    "mps" if torch.backends.mps.is_available() else "cuda"
                ),
            )
            if weight_loss
            else None
        )

    def audio_preparation_hook(self, audio_t):
        """should be overwritten in the child class,
        if audio processing is necessary before passing through the model"""
        return audio_t

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred = self.forward(x)

        # reduce first 2 dimensions
        n_labels = len(self.label_encoder.labels)
        y_target = y_target.view(-1, n_labels)
        y_pred = y_pred.view(-1, n_labels)

        loss = torch.nn.functional.cross_entropy(
            input=y_pred, target=y_target, weight=self.weights
        )
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

        # NOTE - split loss for the first n elements
        n_single = len(
            list(filter(lambda e: e < 2, map(len, self.label_encoder.labels)))
        )
        if (
            self.config.train.validation_metric == "loss"
            or "loss" in self.config.train.extra_val_metrics
        ):
            loss = torch.nn.functional.cross_entropy(
                input=y_pred, target=y_target, weight=self.weights
            )
            self.log(
                "val/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if "partial_loss" in self.config.train.extra_val_metrics:
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
        # loss per label
        if "label_loss" in self.config.train.extra_val_metrics:
            for i in range(n_single):
                weights = torch.zeros(size=(len(self.label_encoder),)).to(self.device)
                weights[i] = 1.0
                label_loss = torch.nn.functional.cross_entropy(
                    input=y_pred, target=y_target, weight=weights
                )
                label_name = " & ".join(self.label_encoder.inv_transform(i))
                self.log(
                    f"val/label_loss_{label_name if label_name else 'NOISE'}",
                    label_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        if (
            self.config.train.validation_metric == "f1_score"
            or "f1_score" in self.config.train.extra_val_metrics
        ):
            f1_score_p_class = multiclass_f1_score(
                preds=y_pred.argmax(-1),
                target=y_target.argmax(-1),
                num_classes=len(self.label_encoder),
                average=None,
                zero_division=0,
            )
            # macro average
            self.log(
                "val/f1_score",
                f1_score_p_class.mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val/partial_f1_score",
                f1_score_p_class[:n_single].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            for i, c_f1_score in enumerate(f1_score_p_class):
                c_f1_score = round(c_f1_score.item(), 6)
                label_name = " & ".join(self.label_encoder.inv_transform(i))
                self.log(
                    f"val/f1_score/{label_name if label_name else 'NOISE'}",
                    c_f1_score,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        if (
            self.config.train.validation_metric == "auroc"
            or "auroc" in self.config.train.extra_val_metrics
        ):
            auroc_per_class = multiclass_auroc(
                preds=y_pred,
                target=y_target.argmax(-1),
                num_classes=len(self.label_encoder),
                average="none",
            )
            self.log(
                "val/auroc",
                auroc_per_class.mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val/partial_auroc",
                auroc_per_class[:n_single].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # ROC curves plotting
        if "roc" in self.config.train.extra_val_metrics:
            fpr_s, tpr_s, _ = multiclass_roc(
                preds=y_pred,
                target=y_target.argmax(-1),
                num_classes=len(self.label_encoder),
            )

            roc_fig = plt.figure(figsize=(10, 5))
            roc_ax = roc_fig.add_subplot()
            for fpr, tpr, label in zip(fpr_s, tpr_s, self.label_encoder.labels):
                labels_str = (
                    " & ".join([e[:3] for e in label]) if label != () else "None"
                )
                roc_ax.plot(
                    fpr.cpu(),
                    tpr.cpu(),
                    label=f"{labels_str} - AUC={round(float(auroc_per_class[self.label_encoder.transform(label)]), 4)}",
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
                # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html
                # self.logger.log_image(key="ROC_curves", images=[roc_fig])
                self.logger.experiment.log({"ROC_curves": wandb.Image(roc_fig)})
            except Exception as _:
                pass
            plt.close(roc_fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
