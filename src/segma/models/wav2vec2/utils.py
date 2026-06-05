from pathlib import Path
from typing import Literal

import torch
from torchaudio.models import wav2vec2_base


def checkpoint_to_w2v2_base_model(
    checkpoint_path: Path | None, device: Literal["cpu", "cuda", "mps"] = "cpu", large : bool = False
):
    """The model was trained and saved using `hubert_pretrain_base` which contains
    specific state for the training part, that we remove here.
    """
    model = wav2vec2_base()
    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError("Please provide a valid path to a checkpoint.")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model
