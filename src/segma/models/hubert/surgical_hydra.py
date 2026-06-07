from typing import Literal

import torch
import torch.nn as nn

from segma.config import ModelConfig
from segma.models.base import ConvolutionSettings
from segma.models.hubert.utils import (
    checkpoint_to_hubert_base_model,
    load_pretrained_hubert_for_finetuning,
)
from segma.utils.encoders import MultiLabelEncoder


class VTC2(nn.Module):
    def __init__(
        self,
        label_encoder: MultiLabelEncoder,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = label_encoder

        self.encoder = checkpoint_to_hubert_base_model(self.config.model_checkpoint)

        # NOTE - freeze CNN encoder
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False

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

    def forward(self, x: torch.Tensor, output: Literal["tensor", "dict"] = "tensor"):
        with torch.no_grad():
            # (batch_size, 199, 512)
            x, lengths = self.encoder.feature_extractor(x, None)
            # (batch_size, 199, 768): 4s
            # (batch_size, 749, 512): 15s
        hidden_states = self.encoder.encoder.extract_features(
            x, lengths, num_layers=None
        )

        x = self.dropout(hidden_states[-1])
        if output == "dict":
            return {name: head(x) for name, head in self.task_heads.items()}
        elif output == "tensor":
            return torch.stack([head(x) for head in self.task_heads.values()], dim=-1)
        else:
            raise ValueError(f"Output type: {output} is not supported.")


class VTC2HuBERT(nn.Module):
    def __init__(
        self,
        label_encoder: MultiLabelEncoder,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = label_encoder

        raw = load_pretrained_hubert_for_finetuning(self.config.model_id)

        # HUBERT_LARGE / HUBERT_XLARGE set `_normalize_waveform=True`, so
        # `bundle.get_model()` returns a `_Wav2Vec2Model` wrapper whose real
        # Wav2Vec2Model is at `.model` and which layer-norms the input
        # waveform before the feature extractor. HUBERT_BASE returns the
        # bare Wav2Vec2Model. Unwrap and remember whether to normalize.
        if hasattr(raw, "model"):  # _Wav2Vec2Model wrapper (large / xlarge)
            self.normalize_waveform = bool(raw.normalize_waveform)
            self.encoder = raw.model
        else:  # bare Wav2Vec2Model (base)
            self.normalize_waveform = False
            self.encoder = raw

        self.encoder_output_size = (
            self.encoder.encoder.feature_projection.projection.out_features
        )

        # NOTE - freeze CNN feature extractor
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout()
        self.task_heads = nn.ModuleDict(
            {
                # encoder_embed_dim
                f"linear_head_{label}": nn.Linear(
                    in_features=self.encoder_output_size, out_features=1
                )
                for label in label_encoder.base_labels
            }
        )

        self.conv_settings = ConvolutionSettings(
            kernels=(10, 3, 3, 3, 3, 2, 2),
            strides=(5, 2, 2, 2, 2, 2, 2),
            paddings=(0, 0, 0, 0, 0, 0, 0),
        )

    def forward(self, x: torch.Tensor, output: Literal["tensor", "dict"] = "tensor"):
        # Normalize by hand since we bypass `_Wav2Vec2Model.forward`.
        if self.normalize_waveform:
            x = nn.functional.layer_norm(x, x.shape)

        with torch.no_grad():
            x, lengths = self.encoder.feature_extractor(x, None)

        hidden_states = self.encoder.encoder.extract_features(
            x, lengths, num_layers=None
        )

        x = self.dropout(hidden_states[-1])
        if output == "dict":
            return {name: head(x) for name, head in self.task_heads.items()}
        elif output == "tensor":
            return torch.stack([head(x) for head in self.task_heads.values()], dim=-1)
        else:
            raise ValueError(f"Output type: {output} is not supported.")
