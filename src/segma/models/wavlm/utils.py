from pathlib import Path

from transformers.models.wavlm.modeling_wavlm import WavLMModel


def load_wavlm(path: Path | str):
    """loads the local wavlm encoder and returns the feature extractor and the encoder."""
    model = WavLMModel.from_pretrained(path, local_files_only=True)
    for param in model.parameters():
        param.requires_grad = False
    model.freeze_feature_encoder()
    return model.feature_extractor, model
