import argparse
from pathlib import Path

import torch

from segma.models import PyanNet, Whisperidou
from segma.predict import prediction
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    l_encoder = PowersetMultiLabelEncoder(
        ["male", "female", "key_child", "other_child"]
    )

    # TODO - argparser to load a folder of wav
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs", default="data/debug/wav")
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        default="models/last/best.ckpt",
        help="Path to a pretrained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        help="Path to a pretrained model checkpoint.",
    )

    args = parser.parse_args()
    args.wavs = Path(args.wavs)
    args.ckpt = Path(args.ckpt)

    assert args.wavs.exists()
    assert args.ckpt.exists()

    # NOTE - resolve output_path
    # if path is model/last/best -> resolve symlink
    if args.output is None and str(args.ckpt) == "models/last/best.ckpt":
        args.output = Path("/".join(args.ckpt.resolve().parts[-4:-2]))
        print(args.output)
    else:
        args.output = Path("segma_out")
    exit(55)

    # model = PyanNet.load_from_checkpoint(
    model = Whisperidou.load_from_checkpoint(
        checkpoint_path=args.ckpt, label_encoder=l_encoder
    )

    model.to(torch.device("mps"))

    for wav_f in args.wavs.glob("*.wav"):
        print(f"[log] - running inference for file: '{wav_f.stem}'")
        prediction(wav_f, model=model, output_p=args.output)
