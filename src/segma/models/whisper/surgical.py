import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder

from .utils import load_whisper


class SurgicalWhisper(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ) -> None:
        super().__init__(label_encoder, config, weight_loss)

        # NOTE - whisper encoder stack
        self.feature_extractor, self.w_encoder = load_whisper(
            self.config.model.config.encoder
        )

        # NOTE - get list of encoder layers to use
        if (
            config.model.config.encoder_layers is None
            or len(config.model.config.encoder_layers) == 0
        ):
            self.enc_layers_to_use = list(
                range(self.w_encoder.config.num_hidden_layers)
            )
        else:
            self.enc_layers_to_use = sorted(
                [i - 1 for i in config.model.config.encoder_layers]
            )

        # NOTE - get redction mechanism and init weights
        if config.model.config.reduction == "weighted":
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use)
            )
        elif config.model.config.reduction == "average":
            # non-learnable weights (buffer) for simple averaging
            self.register_buffer(
                "layer_weights",
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use),
            )
        else:
            raise ValueError(
                f"Should not happen, `{self.config.model.config.reduction=}` should be 'average' or 'weighted'."
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.w_encoder.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, len(label_encoder.labels)),
        )

        self.conv_settings = ConvolutionSettings(
            kernels=(400, 3, 3), strides=(160, 1, 2), paddings=(200, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        enc_x: BaseModelOutput = self.w_encoder(x, output_hidden_states=True)
        # NOTE - first element is the output of the embeddings (conv output)
        hidden_states = enc_x.hidden_states[1:]
        # (n, batch, 1500, enc.dim)
        hidden_states_t = torch.stack(
            [hidden_states[i] for i in self.enc_layers_to_use], dim=0
        )

        if self.config.model.config.reduction == "weighted":
            # Apply softmax to weights for weighted average
            weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
        else:
            # Use uniform weights for simple average
            weights = self.layer_weights

        # Compute average (weighted or simple)
        # (batch, 1500, enc.dim)
        weighted_t = torch.einsum(
            # "l, lbsh->bsh",
            "l,l...->...",
            weights,
            hidden_states_t,
        )

        # NOTE - classic
        logits = self.classifier(weighted_t)

        truncation_i = self.conv_settings.n_windows(
            self.config.audio.chunk_duration_f, strict=False
        )
        logits = logits[:, :truncation_i, :]
        return torch.nn.functional.softmax(logits, dim=-1)

    def audio_preparation_hook(self, audio_t):
        # 'np': numpy | 'pt': pytorch
        return self.feature_extractor(
            audio_t, return_tensors="pt", sampling_rate=self.config.audio.sample_rate
        )["input_features"]
