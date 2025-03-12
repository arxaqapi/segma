# import torch
# import torch.nn as nn
# from transformers.models.wavlm.modeling_wavlm import WavLMEncoder, WavLMFeatureExtractor

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.utils.encoders import LabelEncoder, MultiLabelEncoder


class HydraWave(BaseSegmentationModel):
    def __init__(
        self, label_encoder: LabelEncoder, config: Config, weight_loss: bool
    ) -> None:
        raise NotImplementedError
        assert isinstance(label_encoder, MultiLabelEncoder)
        #         super().__init__(label_encoder, config, weight_loss)

        #         # NOTE - load wavlm encoder
        #         # NOTE - setup backbone
        #         self.lstm_shared = nn.LSTM(
        #             input_size=self.w_encoder.config.d_model,
        #             **config.model.config.lstm.as_dict(),
        #         )

        #         lstm_out_features = (
        #             self.lstm_shared.hidden_size * 2
        #             if self.lstm_shared.bidirectional
        #             else self.lstm_shared.hidden_size
        #         )
        #         # NOTE - setup classification heads
        #         self.task_heads = nn.ModuleDict(
        #             {
        #                 f"linear_head_{label}": nn.Linear(lstm_out_features, 1)
        #                 for label in label_encoder.base_labels
        #             }
        #         )

        self.conv_settings = ConvolutionSettings(
            kernels=(10, 3, 3, 3, 3, 2, 2), strides=(5, 2, 2, 2, 2, 2, 2), paddings=()
        )


#     def forward(self, x):
#         pass

#     def audio_preparation_hook(self, audio_t):
#         return super().audio_preparation_hook(audio_t)
