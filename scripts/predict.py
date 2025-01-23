import argparse
from pathlib import Path

import torch

from segma.config import Config, load_config
from segma.models import Models
from segma.predict import sliding_prediction

# from segma.predict import prediction
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/segma/config/default.yml",
        help="Config file to be loaded and used for inference.",
    )
    parser.add_argument("--uris", help="list of uris to use for prediction")
    parser.add_argument("--wavs", default="data/debug/wav")
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        default="models/last/best.ckpt",
        help="Path to a pretrained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        help="Output Path to the folder that will contain the final predictions.",
    )

    args = parser.parse_args()
    args.wavs = Path(args.wavs)
    args.ckpt = Path(args.ckpt)

    if not args.wavs.exists():
        raise ValueError(f"Path `{args.wavs=}` does not exists")
    if not args.ckpt.exists():
        raise ValueError(f"Path `{args.ckpt=}` does not exists")

    cfg: Config = load_config(args.config)

    l_encoder = PowersetMultiLabelEncoder(labels=cfg.data.classes)

    # NOTE - resolve output_path
    # if path is model/last/best -> resolve symlink
    if args.output is None and str(args.ckpt) == "models/last/best.ckpt":
        args.output = Path("/".join(args.ckpt.resolve().parts[-4:-2]))
    elif args.output is None:
        try:
            args.output = Path("/".join(args.ckpt.parts[-4:-2]))
        except:
            args.output = Path("segma_out")

    # REVIEW
    model = Models[cfg.train.model.name].load_from_checkpoint(
        checkpoint_path=args.ckpt, label_encoder=l_encoder, config=cfg
    )

    model.to("mps" if torch.backends.mps.is_available() else "cuda")

    # NOTE if args.uris: path is known
    if args.uris:
        with Path(args.uris).open("r") as uri_f:
            uris = [uri.strip() for uri in uri_f.readlines()]
        for uri in uris:
            wav_f = (args.wavs / uri).with_suffix(".wav")
            print(f"[log] - running inference for file: '{wav_f.stem}'")
            # prediction(wav_f, model=model, output_p=args.output)
            sliding_prediction(wav_f, model=model, output_p=args.output, config=cfg)
    else:
        for wav_f in args.wavs.glob("*.wav"):
            print(f"[log] - running inference for file: '{wav_f.stem}'")
            # prediction(wav_f, model=model, output_p=args.output)
            sliding_prediction(wav_f, model=model, output_p=args.output, config=cfg)

    # NOTE - symlink to models/last/[rttm|aa]
    static_p = Path("models/last")
    static_p.mkdir(parents=True, exist_ok=True)
    rttm_static_p = static_p / "rttm"
    aa_static_p = static_p / "aa"

    rttm_static_p.unlink(missing_ok=True)
    rttm_static_p.symlink_to((args.output / "rttm").absolute())

    aa_static_p.unlink(missing_ok=True)
    aa_static_p.symlink_to((args.output / "aa").absolute())
