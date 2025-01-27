from pathlib import Path

import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder


class Miniseg(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool
    ) -> None:
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )

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


class Minisinc(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool
    ) -> None:
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )

        self.net = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=251, stride=10, padding=0),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(80, affine=True),
            nn.LeakyReLU(),
            nn.Conv1d(80, 60, 5, stride=1),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(60, affine=True),
            nn.LeakyReLU(),
            nn.Conv1d(60, 60, 5, stride=1),
            nn.MaxPool1d(3, stride=3, padding=0),
            nn.InstanceNorm1d(60, affine=True),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(60, len(label_encoder.labels))

        self.conv_settings = ConvolutionSettings(
            kernels=(251, 3, 5, 3, 5, 3), strides=(10, 3, 1, 3, 1, 3), paddings=(0,) * 6
        )

    def forward(self, x):
        x = x[:, None, :]
        x = self.net(x)
        x = x.transpose(2, 1)
        logits = self.classifier(x)
        return torch.nn.functional.softmax(logits, dim=-1)


class Whisperidou(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ) -> None:
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )
        self.save_hyperparameters(self.config.train.model.config.as_dict())

        self.feature_extractor = WhisperFeatureExtractor()

        self.w_encoder = WhisperEncoder.from_pretrained(
            self.config.train.model.config.encoder, local_files_only=True
        )
        self.w_encoder._freeze_parameters()

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
            self.config.audio_config.chunk_duration_f, strict=False
        )
        logits = logits[:, :truncation_i, :]
        return torch.nn.functional.softmax(logits, dim=-1)

    def audio_preparation_hook(self, audio_t):
        # 'np': numpy | 'pt': pytorch
        return self.feature_extractor(
            audio_t, return_tensors="pt", sampling_rate=16_000
        )["input_features"]


class WhisperiMax(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ) -> None:
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )
        self.save_hyperparameters(self.config.train.model.config.as_dict())

        self.feature_extractor = WhisperFeatureExtractor()

        self.w_encoder = WhisperEncoder.from_pretrained(
            self.config.train.model.config.encoder, local_files_only=True
        )
        self.w_encoder._freeze_parameters()

        self.lstm = nn.LSTM(
            input_size=self.w_encoder.config.d_model,
            **self.config.train.model.config.lstm.as_dict(),
        )

        lstm_out_features: int = self.lstm.hidden_size * 2

        self.linear = nn.Sequential(
            nn.Linear(lstm_out_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(128, len(label_encoder.labels))

        self.conv_settings = ConvolutionSettings(
            kernels=(400, 3, 3), strides=(160, 1, 2), paddings=(200, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        enc_x: BaseModelOutput = self.w_encoder(x).last_hidden_state
        # Since whisper expects 30s audio segments as input (480_000 frames)
        # we have to truncate the output to only cover 2s of audio
        truncation_i = self.conv_settings.n_windows(
            self.config.audio_config.chunk_duration_f, strict=False
        )
        enc_x = enc_x[:, :truncation_i, :]

        lstm_out, _ = self.lstm(enc_x)
        linear_out = self.linear(lstm_out)

        logits = self.classifier(linear_out)

        return torch.nn.functional.softmax(logits, dim=-1)

    def audio_preparation_hook(self, audio_t):
        # 'np': numpy | 'pt': pytorch
        return self.feature_extractor(
            audio_t, return_tensors="pt", sampling_rate=16_000
        )["input_features"]


def load_whisper(path: Path | str):
    feature_extractor = WhisperFeatureExtractor()
    w_encoder = WhisperEncoder.from_pretrained(path, local_files_only=True)
    w_encoder._freeze_parameters()
    return feature_extractor, w_encoder


class SurgicalWhisper(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ) -> None:
        super().__init__(label_encoder, config, weight_loss)

        # NOTE - whisper encoder stack
        self.feature_extractor, self.w_encoder = load_whisper(
            self.config.train.model.config.encoder
        )

        # NOTE - get list of encoder layers to use
        if (
            config.train.model.config.encoder_layers is None
            or len(config.train.model.config.encoder_layers) == 0
        ):
            self.enc_layers_to_use = list(
                range(self.w_encoder.config.num_hidden_layers)
            )
        else:
            self.enc_layers_to_use = sorted(
                [i - 1 for i in config.train.model.config.encoder_layers]
            )

        # NOTE - get redction mechanism and init weights
        if config.train.model.config.reduction == "weighted":
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use)
            )
        elif config.train.model.config.reduction == "average":
            # non-learnable weights (buffer) for simple averaging
            self.register_buffer(
                "layer_weights",
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use),
            )
        else:
            raise ValueError(
                f"Should not happen, `{self.config.train.model.config.reduction=}` should be 'average' or 'weighted'."
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

        if self.config.train.model.config.reduction == "weighted":
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
            self.config.audio_config.chunk_duration_f, strict=False
        )
        logits = logits[:, :truncation_i, :]
        return torch.nn.functional.softmax(logits, dim=-1)

    def audio_preparation_hook(self, audio_t):
        # 'np': numpy | 'pt': pytorch
        return self.feature_extractor(
            audio_t, return_tensors="pt", sampling_rate=16_000
        )["input_features"]


class HydraWhisper(BaseSegmentationModel):
    def __init__(self, label_encoder, config, weight_loss=False):
        super().__init__(label_encoder, config, weight_loss)
        raise NotImplementedError
