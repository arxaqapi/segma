from pathlib import Path
import torch
from torch.nn import Module
from transformers.models.wavlm.modeling_wavlm import WavLMModel
import torchaudio
from torchaudio.models import hubert_pretrain_base
from segma.models.wav2vec.fairseq_wav2vec import FairseqWav2Vec2


def load_wav2vec(path: Path | str):
    path = Path(path)

    # TODO bad habitude to load with num_clusters fixed
    model = FairseqWav2Vec2(path, output_all_hiddens=True, tgt_layer=None)
    return model

