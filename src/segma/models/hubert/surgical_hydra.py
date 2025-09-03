from typing import Any, Mapping

import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_f1_score

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder, MultiLabelEncoder

from .utils import load_hubert


# NOTE - Heavily copied from whisper/hydra.py
class SurgicalHydraHubert(BaseSegmentationModel):
    def __init__(
        self,
        label_encoder: LabelEncoder,
        config: Config,
        weight_loss: bool = False,
        loss_f: str = "bce",
    ) -> None:
        super().__init__(label_encoder, config, weight_loss)
        if not isinstance(label_encoder, MultiLabelEncoder):
            raise ValueError("Only MultiLabelEncoder is accepted for HydraWavLM.")

        self.wav2vec2, self.encoder = load_hubert(self.config.model.config.wav_encoder)

        if not config.train.lstm :
            self.wav2vec2.feature_extractor.eval()
            self.wav2vec2.feature_extractor._require_grad = False
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
            
        if (
            config.model.config.encoder_layers is None
            or len(config.model.config.encoder_layers) == 0
        ):
            self.enc_layers_to_use = list(
                range(len(self.encoder.wav2vec2.encoder.transformer.layers))
            )
        else:
            self.enc_layers_to_use = sorted(
                [i - 1 for i in config.model.config.encoder_layers]
            )
        # reduction mechanism - learnable or non-learnable weights
        if config.model.config.reduction == "weighted":
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use)
            )
        elif config.model.config.reduction == "average":
            self.register_buffer(
                "layer_weights",
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use),
            )
        elif not config.train.lstm: 
            pass
        else:
            raise ValueError(
                f"Should not happen, `{self.config.model.config.reduction=}` should be `average` or `weighted`"
            )

        self.sequence_classifier = "speech-maturity" in config.data.dataset_path

        if config.train.lstm :
            # TODO bad habit dimension fixed
            self.lstm_shared = nn.LSTM(
                input_size=768,
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
        else:
            self.dropout = nn.Dropout()
            self.task_heads = nn.ModuleDict(
                {
                    f"linear_head_{label}": nn.Linear(in_features=768, out_features=1)
                    for label in label_encoder.base_labels
                }
            )

        self.conv_settings = ConvolutionSettings(
            kernels=(10, 3, 3, 3, 3, 2, 2),
            strides=(5, 2, 2, 2, 2, 2, 2),
            paddings=(0, 0, 0, 0, 0, 0, 0),
        )

        # NOTE - copied from whisper/hydra.py

    def forward(self, x: torch.Tensor):
        # TODO
       
        x, lengths = self.wav2vec2.feature_extractor(x, None)
        hidden_states = self.wav2vec2.encoder.extract_features(x, lengths, num_layers=None)
        # enc_x: BaseModelOutput = self.encoder(x, output_hidden_states=True)
        # hidden_states = enc_x.hidden_states[1:]
        if self.config.train.lstm :
            hidden_states_t = torch.stack(
                [hidden_states[i] for i in self.enc_layers_to_use], dim=0
            )

            # NOTE - handle weighted encoder output
            if self.config.model.config.reduction == "weighted":
                weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
            else:
                # uniform weights for simple average
                weights = self.layer_weights

            weighted_t = torch.einsum(
                "l,l...->...",
                weights,
                hidden_states_t,
            )

            # take the weighteds and convolve them from (B, S, hidden) to (B,1,hidden) here
            if False:  # self.sequence_classifier:
                x = weighted_t.transpose(1, 2)
                x = self.conv1(x)
                x = self.conv2(x)
                x = x.transpose(1, 2)  # (B, 28, 768)
                x = self.norm(x)
                weights = self.attn_conv(x).softmax(dim=1)
                x = (x * weights).sum(dim=1, keepdim=True)
                lstm_out, _ = self.lstm_shared(x)

            # (batch, 99, lstm.hidden_size)
            else:
                lstm_out, _ = self.lstm_shared(weighted_t)
            return {
                name: head(lstm_out)
                # NOTE - sigmoid is added in BCEWithLogitsLoss, return only logits
                # name: nn.functional.sigmoid(head(lstm_out))
                for name, head in self.task_heads.items()
            }
        else:
            x = self.dropout(hidden_states[-1])
            #print("x shape", x.shape)
            return {
                
                name: head(x)
                # NOTE - sigmoid is added in BCEWithLogitsLoss, return only logits
                # name: nn.functional.sigmoid(head(lstm_out))
                for name, head in self.task_heads.items()
            } 

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        # if we want to repeat the targets to make VTC like predictions
        # y_target = y_target.repeat(1,28,1)
        y_pred_heads = self.forward(x)
        # add sigmoid here

        # 'linear_head_male', 'linear_head_female',
        # 'linear_head_key_child', 'linear_head_other_child'
        # batch and windows merged, maybe not because no window
        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.labels)
        # (batch * n_windows, 4)
        y_target = y_target.view(-1, n_labels)  # .repeat_interleave(28)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}

        # FIXME - y_target should be one value per label
        head_losses = {
            k: torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred, target=y_target[..., i], weight=self.weights
            )
            for i, (k, y_pred) in enumerate(y_preds.items())
        }
        # average pooling 28 x N
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
                logger=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        # repeat labels for vtc like prediction
        # y_target = y_target.repeat(1,28,1)
        y_pred_heads = self.forward(x)
        # add sigmoid here

        # 'linear_head_male', 'linear_head_female', 'linear_head_key_child', 'linear_head_other_child'

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.labels)
        # (batch * n_windows, 4)
        y_target = y_target.view(-1, n_labels)  # .repeat_interleave(28)
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
                logger=True
            )
            for head_name, head_loss in head_losses.items():
                self.log(
                    f"val/loss_{head_name.removeprefix('linear_head_')}",
                    head_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True
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
                    logger=True

                )
            # TODO - total f1_score using a merger of TP/FP ...

    def state_dict(self, *args, **kwargs):
        """Custom state_dict that excludes the whisper encoder."""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all entries starting with 'w_encoder.'
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("encoder.")]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Custom load_state_dict that doesn't require whisper encoder weights."""
        return super().load_state_dict(state_dict, strict=False, assign=assign)
