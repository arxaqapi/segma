from dataclasses import dataclass
from functools import cached_property, reduce
from math import floor, prod

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


@dataclass
class ConvolutionSettings:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]

    def __post_init__(self):
        if not (len(self.kernels) == len(self.strides) == len(self.paddings)):
            raise ValueError(
                "Given settings do not match, please provide matching dimensions for kernels, strides and paddings."
            )

    def rf_start_i(self, u_L: int) -> int:
        """Computes the start index of the receptive field.

        see eq (5) in https://distill.pub/2019/computing-receptive-fields/

        Args:
            u_L (int): start index of the output range.

        Returns:
            int: Start index of the receptive field in the input vector. Can be negative.
        """
        L = len(self.strides)
        assert L == len(self.paddings)
        S_0 = prod(self.strides)

        P_0 = 0
        for layer_i in range(L):
            P_0 += self.paddings[layer_i] * prod(self.strides[:layer_i])

        return u_L * S_0 - P_0

    def rf_end_i(self, v_L: int) -> int:
        """Computes the end index of the receptive field.

        see eq (6) in https://distill.pub/2019/computing-receptive-fields/

        Args:
            v_L (int): end index of the output range.

        Returns:
            int: End index of the receptive field in the input vector. Can be greater than the size of the input vector.
        """
        L = len(self.kernels)
        assert L == len(self.strides) == len(self.paddings)

        S_0 = prod(self.strides)

        rt = 0
        for layer_i in range(L):
            rt += (1 + self.paddings[layer_i] - self.kernels[layer_i]) * prod(
                self.strides[:layer_i]
            )

        return v_L * S_0 - rt

    @cached_property
    def rf_size(self) -> int:
        """Computes the size of the receptive field.

        see eq (2) in https://distill.pub/2019/computing-receptive-fields/

        Returns:
            int: Size of the receptive field.
        """
        L = len(self.kernels)
        assert L == len(self.strides)

        rf = 0
        for layer_i in range(L):
            rf += (self.kernels[layer_i] - 1) * prod(self.strides[:layer_i])
        return rf + 1

    def rf_center_i(self, u_L: int):
        """Center of receptive field"""
        L = len(self.kernels)
        assert L == len(self.strides) == len(self.paddings)

        S_0 = prod(self.strides)
        P_0 = 0
        for layer_i in range(L):
            P_0 += self.paddings[layer_i] * prod(self.strides[:layer_i])

        return u_L * S_0 + (self.rf_size - 1) / 2 - P_0

    @cached_property
    def rf_step(self) -> int:
        """Returns the step size (stride) between 2 receptive fields.

        Returns:
            int: step size/stride between 2 receptive fields.
        """
        assert (
            abs(self.rf_start_i(0) - self.rf_start_i(1))
            == abs(self.rf_end_i(0) - self.rf_end_i(1))
            == abs(self.rf_center_i(0) - self.rf_center_i(1))
        )
        return abs(self.rf_start_i(0) - self.rf_start_i(1))

    def n_windows(self, chunk_duration_f: int, strict: bool = True) -> int:
        """Compute the total number of convolution windows that can fit in a given audio chunk.

        Args:
            chunk_duration_f (int): Duration of the audio chunk in frames.
            strict (bool, optional):
                If True, only count windows that fully fit within the chunk.
                If False, allow windows that partially exceed the chunk. Defaults to True.

        Returns:
            int: Number of valid convolution windows.
        """
        # Add a correction if any kernel has even size (can affect center alignment)
        has_even_kernel = reduce(lambda b, e: b or (e % 2 == 0), self.kernels, False)
        correction = 1 if has_even_kernel else 0

        # Should be 320 (f) with duration 2 secs and whisper model
        # Should be 270 (f) with duration 2 secs and sinc model
        rf_step = int(self.rf_step + correction)

        if strict:
            return floor((chunk_duration_f - self.rf_size) / rf_step) + 1
        else:
            return chunk_duration_f // rf_step


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
        self.conv_settings = ConvolutionSettings((0,), (0,), (0,))

        self.save_hyperparameters(self.config.as_dict())

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
