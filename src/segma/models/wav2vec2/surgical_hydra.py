from typing import Literal

import torch
import torch.nn as nn

from segma.config import ModelConfig
from segma.models.base import ConvolutionSettings
from segma.models.wav2vec2.utils import checkpoint_to_w2v2_base_model
from segma.utils.encoders import MultiLabelEncoder


class W2V4300LL(nn.Module):
    def __init__(
        self,
        label_encoder: MultiLabelEncoder,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = label_encoder
        self.hidden_dim = 768
        
        self.encoder = checkpoint_to_w2v2_base_model(self.config.model_checkpoint)   

        # NOTE - freeze CNN encoder
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout()
        self.task_heads = nn.ModuleDict(
            {
                f"linear_head_{label}": nn.Linear(in_features=self.hidden_dim, out_features=1)
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
            ValueError(f"Output type: {output} is not supported.")
