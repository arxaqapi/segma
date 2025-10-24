from pathlib import Path

import torch
import torchaudio
from torch.nn import Module
from torchaudio.models import hubert_pretrain_base
from transformers.models.wavlm.modeling_wavlm import WavLMModel


def load_wavlm(path: Path | str):
    """loads the local wavlm encoder and returns the feature extractor and the encoder."""
    model = WavLMModel.from_pretrained(path, local_files_only=True)
    for param in model.parameters():
        param.requires_grad = False
    model.freeze_feature_encoder()
    return model.feature_extractor, model


def load_hubert(path: Path | str):
    path = Path(path)

    model = hubert_pretrain_base(num_classes=500)
    if path.exists():
        model = _load_state(model, path)
    else:
        bundle = torchaudio.pipelines.HUBERT_BASE
        wav2vec2 = bundle.get_model()
        model.wav2vec2 = wav2vec2

    return model.wav2vec2


def _load_state(model: Module, checkpoint_path: Path, device="cpu") -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model
