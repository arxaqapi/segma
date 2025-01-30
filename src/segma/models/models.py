import torch
import torch.nn as nn

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
