from pathlib import Path
from segma.models.wav2vec.fairseq_wav2vec import FairseqWav2Vec2


def load_wav2vec(path: Path | str):
    # TODO bad habitude to load with num_clusters fixed
    model = FairseqWav2Vec2(path, output_all_hiddens=True, tgt_layer=None, freeze=False)
    return model

