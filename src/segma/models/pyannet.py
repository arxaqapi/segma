import itertools

import torch
import torch.nn as nn
from asteroid_filterbanks import Encoder, ParamSincFB
from torchmetrics.functional.classification import multiclass_auroc, multiclass_f1_score

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder  # PowersetMultiLabelEncoder


class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")

        self.sample_rate = sample_rate
        self.stride = stride

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=self.stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """

        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            if c == 0:
                outputs = torch.abs(outputs)

            outputs = torch.nn.functional.leaky_relu(norm1d(pool1d(outputs)))

        return outputs


class PyanNet(BaseSegmentationModel):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        config: Config,
        weight_loss: bool = False,
    ):
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )

        sincnet = self.config.model.config.sincnet
        lstm_c = self.config.model.config.lstm

        self.save_hyperparameters(self.config.model.config.as_dict())

        self.sincnet = SincNet(
            sample_rate=self.config.audio_config.sample_rate, stride=sincnet.stride
        )

        self.lstm = nn.LSTM(
            input_size=60,
            hidden_size=lstm_c.hidden_size,
            num_layers=lstm_c.num_layers,
            batch_first=True,
            bidirectional=lstm_c.bidirectional,
            dropout=lstm_c.dropout,
        )

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in itertools.pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear[0]] * len(self.hparams.linear)
                )
            ]
        )
        self.classifier = nn.Linear(
            self.hparams.classifier, len(self.label_encoder.labels)
        )
        self.activation = nn.Sigmoid()

        # Sincnet + pyannet
        self.conv_settings = ConvolutionSettings(
            kernels=(251, 3, 5, 3, 5, 3),
            strides=(self.sincnet.stride, 3, 1, 3, 1, 3),
            paddings=(0, 0, 0, 0, 0, 0),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        # adjust dimensions
        waveforms = waveforms[:, None, :]
        outputs = self.sincnet(waveforms)

        outputs, _ = self.lstm(
            # (batch, feature, frame) -> (batch, frame, feature)
            outputs.transpose(2, 1)
        )

        if len(self.hparams.linear) > 0:
            for linear in self.linear:
                outputs = torch.nn.functional.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))


class PyanNetSlim(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool = False
    ):
        super().__init__(
            label_encoder=label_encoder, config=config, weight_loss=weight_loss
        )

        sincnet = self.config.model.config.sincnet

        self.save_hyperparameters(self.config.model.config.as_dict())

        self.sincnet = SincNet(
            sample_rate=self.config.audio_config.sample_rate, stride=sincnet.stride
        )

        # self.linear_stack = nn.ModuleList(
        self.linear = nn.ModuleList(
            [
                nn.Linear(60, self.hparams.linear[0]),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(self.hparams.linear[0], self.hparams.linear[1]),
                nn.LeakyReLU(),
                nn.Dropout(),
            ]
        )
        self.classifier = nn.Linear(
            self.hparams.classifier, len(self.label_encoder.labels)
        )
        self.activation = nn.Sigmoid()

        # Sincnet + pyannet
        self.conv_settings = ConvolutionSettings(
            kernels=(251, 3, 5, 3, 5, 3),
            strides=(self.sincnet.stride, 3, 1, 3, 1, 3),
            paddings=(0, 0, 0, 0, 0, 0),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward
        waveforms : (batch, channel, sample)
        scores : (batch, frame, classes)
        """
        # adjust dimensions
        waveforms = waveforms[:, None, :]
        outputs = self.sincnet(waveforms)
        # (batch, feature, frame) -> (batch, frame, feature)
        outputs = outputs.transpose(2, 1)

        for layer in self.linear:
            outputs = layer(outputs)

        return self.activation(self.classifier(outputs))

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

        if self.config.train.validation_metric == "loss":
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
        if self.config.train.validation_metric == "auroc":
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
        if self.config.train.validation_metric == "f1_score":
            f1_score = multiclass_f1_score(
                preds=y_pred,
                target=y_target.argmax(-1),
                num_classes=len(self.label_encoder),
                # average="none",
            )
            self.log(
                "val/f1_score",
                f1_score,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
