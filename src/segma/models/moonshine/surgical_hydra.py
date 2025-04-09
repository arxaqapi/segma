from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings


class Luna(BaseSegmentationModel):
    def __init__(self, config: Config, weight_loss: bool = False) -> None:
        super().__init__(config, weight_loss)
        raise NotImplementedError

        # NOTE - load moonshine encoder
        #   load, forward, (no)
        # NOTE - setup backbone and head classifier
        self.conv_settings = ConvolutionSettings(
            kernels=(127, 7, 3),
            strides=(64, 3, 2),
            paddings=(0, 0, 0),
        )

    def forward(self, x):
        raise NotImplementedError
        # truncation should not be needed with moonshine since it handles short audio segments
        _truncation_i = self.conv_settings.n_windows(
            self.config.audio.chunk_duration_f, strict=False
        )
