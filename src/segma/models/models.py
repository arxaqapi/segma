import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder


class Miniseg(BaseSegmentationModel):
    def __init__(self, label_encoder: LabelEncoder, config: Config) -> None:
        super().__init__(label_encoder=label_encoder, config=config)

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
    def __init__(self, label_encoder: LabelEncoder, config: Config) -> None:
        super().__init__(label_encoder=label_encoder, config=config)

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
    def __init__(self, label_encoder: LabelEncoder, config: Config) -> None:
        super().__init__(label_encoder=label_encoder, config=config)

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
    def __init__(self, label_encoder: LabelEncoder, config: Config) -> None:
        super().__init__(label_encoder=label_encoder, config=config)

        self.feature_extractor = WhisperFeatureExtractor()

        self.w_encoder = WhisperEncoder.from_pretrained(
            self.config.train.model.config.encoder, local_files_only=True
        )
        self.w_encoder._freeze_parameters()

        self.lstm = nn.LSTM(
            input_size=self.w_encoder.config.d_model,
            hidden_size=128,
            num_layers=4,
            bidirectional=True,
            dropout=0.5,
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
