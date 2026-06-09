from pathlib import Path
from typing import Literal

import torch
from torchaudio.models import hubert_base, hubert_large, hubert_xlarge
from torchaudio.pipelines import HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE


def checkpoint_to_hubert_base_model(
    checkpoint_path: Path | None,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    hubert_size: Literal["base", "large", "xlarge"] = "base",
):
    """The model was trained and saved using `hubert_pretrain_base` which contains
    specific state for the training part, that we remove here.
    """
    match hubert_size:
        case "base":
            model = hubert_base()
        case "large":
            model = hubert_large()
        case "xlarge":
            model = hubert_xlarge()
        case _:
            raise ValueError(f"Argument `hubert_size` is incorrect: {hubert_size}")

    if checkpoint_path is None:
        return model
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError("Please provide a valid path to a checkpoint.")

    expected_model_keys = model.state_dict().keys()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        k.removeprefix("model.").removeprefix("wav2vec2."): v
        for k, v in checkpoint["state_dict"].items()
    }
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in expected_model_keys
    }

    model.load_state_dict(filtered_state_dict)
    return model


def load_pretrained_hubert_for_finetuning(
    hubert_size: Literal["hubert_base", "hubert_large", "hubert_xlarge"] = "base",
):
    """Load a pretrained model, as per the original publication.
    For more informations read the torchaudio documentation.

    - https://docs.pytorch.org/audio/stable/pipelines.html#wav2vec-2-0-hubert-wavlm-ssl
    """

    match hubert_size:
        case "hubert_base":
            bundle = HUBERT_BASE
        case "hubert_large":
            bundle = HUBERT_LARGE
        case "hubert_xlarge":
            bundle = HUBERT_XLARGE
        case _:
            raise ValueError(f"Argument `hubert_size` is incorrect: {hubert_size}")
    return bundle.get_model()
