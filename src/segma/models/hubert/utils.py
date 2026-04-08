from pathlib import Path
from typing import Literal

import torch
from torchaudio.models import hubert_base


def checkpoint_to_hubert_base_model(
    checkpoint_path: Path | None, device: Literal["cpu", "cuda", "mps"] = "cpu"
):
    """The model was trained and saved using `hubert_pretrain_base` which contains
    specific state for the training part, that we remove here.
    """
    if checkpoint_path is None:
        return hubert_base()
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError("Please provide a valid path to a checkpoint.")

    model = hubert_base()
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
