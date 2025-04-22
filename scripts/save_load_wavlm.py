import argparse

import torch
from transformers import WavLMModel

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
            "base",
            "large",
        ),
        default="base",
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

    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-" + args.model,
        torch_dtype=t_type,
    )
    model.save_pretrained(f"wavlm_{args.model.replace('-', '_')}_model")
