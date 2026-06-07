from segma.config import ModelConfig

from .hubert.surgical_hydra import VTC2, VTC2HuBERT
from .multilabel import MultiLabelModel
from .wav2vec2.surgical_hydra import W2V24300LL

__all__ = ["MultiLabelModel", "VTC2", "W2V24300LL"]


def model_resolver(config: ModelConfig):
    """Valid models are: vtc2, hubert_base, hubert_large and w2v2-ll4300"""
    match config.model_id:
        case "vtc2":
            return VTC2
        case "hubert_base":
            return VTC2HuBERT
        case "hubert_large":
            return VTC2HuBERT
        case "w2v2-ll4300":
            return W2V24300LL
        case _:
            raise ValueError("Model is unsuported, please select a valid one")
