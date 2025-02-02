from dataclasses import dataclass

import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_f1_score
from transformers.modeling_outputs import BaseModelOutput

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder, MultiLabelEncoder

from .utils import load_whisper


@dataclass
class HydraOutput:
    pass


class HydraWhisper(BaseSegmentationModel):
    """Model that transforms a multi-class classification problem into
    a multi-task instance, where each task is one of the classes.

    - [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
    - https://en.wikipedia.org/wiki/Multi-task_learning
    - https://www.ruder.io/multi-task/
    - https://discuss.pytorch.org/t/single-class-classification-image/139587
    - https://discuss.pytorch.org/t/multiple-loss-function-optimization/184515
    - https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
    - [Multi-Head Multi-Loss Model Calibration](https://arxiv.org/abs/2303.01099)
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        config: Config,
        weight_loss: bool = False,
        loss_f: str = "bce",
    ) -> None:
        super().__init__(label_encoder, config, weight_loss)
        if not isinstance(label_encoder, MultiLabelEncoder):
            raise ValueError("Only MultiLabelEncoder is accepted for HydraWhisper.")

        self.feature_extractor, self.w_encoder = load_whisper(
            self.config.model.config.encoder
        )

        self.lstm_shared = nn.LSTM(
            input_size=self.w_encoder.config.d_model,
            **config.model.config.lstm.as_dict(),
        )

        lstm_out_features = (
            self.lstm_shared.hidden_size * 2
            if self.lstm_shared.bidirectional
            else self.lstm_shared.hidden_size
        )

        # NOTE - define x heads, one per label
        self.task_heads = nn.ModuleDict(
            {
                f"linear_head_{label}": nn.Linear(lstm_out_features, 1)
                for label in label_encoder.base_labels
            }
        )

        self.conv_settings = ConvolutionSettings(
            kernels=(400, 3, 3), strides=(160, 1, 2), paddings=(200, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        enc_x: BaseModelOutput = self.w_encoder(x).last_hidden_state
        # Since whisper expects 30s audio segments as input (480_000 frames)
        # we have to truncate the output to only cover 2s of audio
        truncation_i = self.conv_settings.n_windows(
            self.config.audio.chunk_duration_f, strict=False
        )
        enc_x = enc_x[:, :truncation_i, :]

        # (batch, 99, lstm.hidden_size)
        lstm_out, _ = self.lstm_shared(enc_x)
        return {
            name: head(lstm_out)
            # NOTE - sigmoid is added in BCEWithLogitsLoss, return only logits
            # name: nn.functional.sigmoid(head(lstm_out))
            for name, head in self.task_heads.items()
        }

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred_heads = self.forward(x)
        # 'linear_head_male', 'linear_head_female',
        # 'linear_head_key_child', 'linear_head_other_child'

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.labels)
        # (batch * n_windows, 4)
        y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}

        # FIXME - y_target should be one value per label
        head_losses = {
            k: torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred, target=y_target[..., i], weight=self.weights
            )
            for i, (k, y_pred) in enumerate(y_preds.items())
        }

        loss = torch.stack(list(head_losses.values())).sum()
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        for head_name, head_loss in head_losses.items():
            self.log(
                f"train/loss_{head_name.removeprefix('linear_head_')}",
                head_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred_heads = self.forward(x)
        # 'linear_head_male', 'linear_head_female', 'linear_head_key_child', 'linear_head_other_child'

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.labels)
        # (batch * n_windows, 4)
        y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}

        # NOTE - loss computation
        if (
            self.config.train.validation_metric == "loss"
            or "loss" in self.config.train.extra_val_metrics
        ):
            head_losses = {
                k: torch.nn.functional.binary_cross_entropy_with_logits(
                    input=y_pred, target=y_target[..., i], weight=self.weights
                )
                for i, (k, y_pred) in enumerate(y_preds.items())
            }

            loss = torch.stack(list(head_losses.values())).sum()
            self.log(
                "val/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            for head_name, head_loss in head_losses.items():
                self.log(
                    f"val/loss_{head_name.removeprefix('linear_head_')}",
                    head_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        # NOTE - f1 score
        if (
            self.config.train.validation_metric == "f1_score"
            or "f1_score" in self.config.train.extra_val_metrics
        ):
            head_f1_scores = {
                k: binary_f1_score(
                    preds=y_pred,
                    target=y_target[..., i],
                    threshold=0.5,
                )
                for i, (k, y_pred) in enumerate(y_preds.items())
            }
            for head_name, head_f1_score in head_f1_scores.items():
                self.log(
                    f"val/f1_{head_name.removeprefix('linear_head_')}",
                    head_f1_score,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            # TODO - total f1_score using a merger of TP/FP ...

        ####################################################################################
        ####################################################################################

    def audio_preparation_hook(self, audio_t):
        # 'np': numpy | 'pt': pytorch
        return self.feature_extractor(
            audio_t, return_tensors="pt", sampling_rate=self.config.audio.sample_rate
        )["input_features"]
