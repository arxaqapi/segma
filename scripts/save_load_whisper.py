import argparse

import torch
from transformers import WhisperModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--precision",
        choices=("fp8", "fp16", "fp32"),
        default="fp16",
        type=str,
        help="Choose the floating point precision of the model weights.",
    )
    parser.add_argument(
        "--model",
        choices=(
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ),
        default="tiny",
        type=str,
        help="Select the model.",
    )

    args = parser.parse_args()

    if args.precision == "fp32":
        t_type = torch.float32
    elif args.precision == "fp16":
        t_type = torch.float16
    elif args.precicion == "fp8":
        t_type = torch.float8_e4m3fn
    else:
        raise ValueError(
            "Should not happen, please select the precision of the model, between ('fp8', 'fp16', 'fp32')"
        )

    model = WhisperModel.from_pretrained(
        "openai/whisper-" + args.model,
        torch_dtype=t_type,
    )
    model.encoder.save_pretrained(f"whisper_{args.model.replace('-', '_')}_encoder")
