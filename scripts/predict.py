import argparse
from pathlib import Path

from segma.models import PyanNet
from segma.predict import prediction
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    l_encoder = PowersetMultiLabelEncoder(
        ["male", "female", "key_child", "other_child"]
    )

    # TODO - argparser to load a folder of wav
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs", default="data/debug/wav")

    args = parser.parse_args()

    args.wavs = Path(args.wavs)

    assert args.wavs.exists()

    for wav_f in args.wavs.glob("*.wav"):
        prediction(wav_f, model=PyanNet(l_encoder))
