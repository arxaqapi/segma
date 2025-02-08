import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder

from .utils import load_whisper


class Whisperidou(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ) -> None:
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )

        self.feature_extractor, self.w_encoder = load_whisper(
            self.config.model.config.encoder
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
        enc_x: BaseModelOutput = self.w_encoder(x)
        logits = self.classifier(enc_x.last_hidden_state)

        # Since whisper expects 30s audio segments as input (480_000 frames)
        # we have to truncate the output to only cover x seconds of audio
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
