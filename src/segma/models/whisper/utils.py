from pathlib import Path

from transformers import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def load_whisper(path: Path | str):
    """loads the local whisper encoder and returns the feature extractor and the encoder."""
    feature_extractor = WhisperFeatureExtractor()
    w_encoder = WhisperEncoder.from_pretrained(path, local_files_only=True)
    w_encoder._freeze_parameters()
    return feature_extractor, w_encoder
