import argparse
from functools import cache
from math import floor
from pathlib import Path

import torch
import torchaudio
import yaml

from segma.config import load_config
from segma.config.base import Config
from segma.models import Models
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.predict import write_intervals
from segma.structs.interval import Intervals
from segma.utils.encoders import MultiLabelEncoder


@cache
def chunk_start_i(i: int) -> int:
    """Start index of chunk `i` while taking into account the missing frames per chunk.

    Args:
        i (int): index of the chunk.

    Returns:
        int: start index of the chunck.
    """
    return i * 199 * 320


@cache
def chunk_end_i(i: int, chunk_duration_f: int) -> int:
    return chunk_start_i(i) + chunk_duration_f


@cache
def chunk_end_i_coverage(i: int) -> int:
    """End of the coverage of chunk `i` taking into account the missing frames skipped by the forward pass of the model.

    Args:
        i (int): index of the chunk.

    Returns:
        int: end index of the chunck.
    """
    return (i + 1) * 199 * 320


@cache
def batch_start_i(i: int, batch_size: int) -> int:
    return i * batch_size * (199 * 320)


@cache
def batch_end_i(i: int, batch_size: int, chunk_duration_f: int) -> int:
    return batch_start_i(i, batch_size) + batch_size * chunk_duration_f


@cache
def batch_end_i_coverage(i: int, batch_size: int, chunk_duration_f: int) -> int:
    return batch_end_i(i, batch_size, chunk_duration_f) - batch_size * 320


def prepare_audio(
    audio_path: Path,
    model: BaseSegmentationModel,
    start_f: int,
    end_f: int | None = None,
):
    """Given the path to an audio file, a model, a start frame and end frame,
    load the corresponding section in the audio, run the `audio_preparation_hook` and put on the right device.

    Args:
        audio_path (Path): Path to the audio file to load.
        model (BaseSegmentationModel): model to use.
        start_f (int): start index in the audio to start loading from.
        end_f (int | None, optional): end index in the audio to end loading from. Defaults to None.

    Returns:
        _type_: _description_
    """
    num_frames = end_f - start_f if end_f else -1
    sub_audio_t = torchaudio.load(
        uri=audio_path.resolve(),
        frame_offset=start_f,
        num_frames=num_frames,
    )[0]
    sub_audio_t = model.audio_preparation_hook(sub_audio_t.squeeze(1)).squeeze(0)
    return sub_audio_t.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def get_n_fitting_chunks(
    n_frames: int, chunk_duration_f: int, missing_n_frames: int
) -> int:
    """Given a number of frames, computes the amount of complete overlaping window that fit in
    the given number of frames.

    Args:
        n_frames (int): number of frames to take into account
        chunk_duration_f (int, optional): Duration or size of the window that should fit entirely. Defaults to 64_000.
        missing_n_frames (int, optional): Missing number of frames or amount of overlap between each window. Defaults to 320.

    Returns:
        int: Number of complete windowd with overlap that fit in `_n_frames`.
    """
    return (
        floor(((n_frames - chunk_duration_f) / (chunk_duration_f - missing_n_frames)))
        + 1
    )


def apply_model_on_audio(
    audio_path: Path,
    model: torch.nn.Module,
    batch_size: int = 128,
    chunk_duration_s: float = 4.0,
    missing_n_frames: int = 320,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Apply model on audio, return tensor of size (n_frames, n_classes)"""
    chunk_duration_f = int(chunk_duration_s * sample_rate)
    n_frames_audio = torchaudio.info(audio_path).num_frames

    n_fitting_chunks = get_n_fitting_chunks(
        n_frames_audio, chunk_duration_f, missing_n_frames
    )
    n_full_batches = floor(n_fitting_chunks / batch_size)

    logits = []
    for i in range(n_full_batches):
        # NOTE - load audio section
        sub_audio_t = prepare_audio(
            audio_path,
            model,
            start_f=batch_start_i(i, batch_size=batch_size),
            end_f=batch_end_i_coverage(i, batch_size, chunk_duration_f)
            + missing_n_frames,
        )

        batch_t = sub_audio_t.unfold(
            0, size=chunk_duration_f, step=chunk_duration_f - missing_n_frames
        )
        if batch_t.shape[0] != batch_size:
            raise ValueError(
                f"Error during unfolding of audio tensor, got '{batch_t.shape[0]}', expected '{batch_size}' elements in the batch"
            )
        # NOTE - pass through model
        with torch.inference_mode():
            out_t = model(batch_t).squeeze(2)
        logits.append(out_t)

    # NOTE - Handle chunks that do not fit in a batch
    leftover_frames = n_frames_audio - batch_end_i_coverage(
        n_full_batches - 1, batch_size, chunk_duration_f
    )
    if leftover_frames:
        # ==========================================
        # NOTE - load audio section
        sub_audio_t = prepare_audio(
            audio_path,
            model,
            start_f=batch_start_i(n_full_batches, batch_size),
            end_f=chunk_start_i(
                n_full_batches * batch_size
                + get_n_fitting_chunks(
                    leftover_frames, chunk_duration_f, missing_n_frames
                )
            )
            + missing_n_frames,
        )
        # NOTE - unfold into chunks
        batch_t = sub_audio_t.unfold(
            0, size=chunk_duration_f, step=chunk_duration_f - missing_n_frames
        )
        # NOTE - pass through model
        with torch.inference_mode():
            out_t = model(batch_t)
        logits.append(out_t)

        # ==========================================
        # NOTE - handle frames that do not fit in a chunk
        last_audio_t = prepare_audio(
            audio_path,
            model,
            start_f=chunk_start_i(
                n_full_batches * batch_size
                + get_n_fitting_chunks(
                    leftover_frames, chunk_duration_f, missing_n_frames
                )
            ),
        )
        # NOTE - pass through model
        with torch.inference_mode():
            out_last_t = model(last_audio_t[None, :])
        logits.append(out_last_t)
        # ==========================================

    return torch.concat(
        [t.view(-1, model.label_encoder.n_labels) for t in logits], dim=0
    )


def apply_thresholds(
    feature_tensor: torch.Tensor, thresholds: dict[str, dict[str, float]]
) -> torch.Tensor:
    """Given raw logits, perform thresholding.

    Args:
        feature_tensor (torch.Tensor): tensor containing the raw logits.
        thresholds (dict[str, dict[str, float]]): threshold dict mapping label to corresponding threshold.

    Returns:
        torch.Tensor: thresholded tensor.
    """
    feature_tensor = feature_tensor.sigmoid()
    assert feature_tensor.shape[-1] == len(thresholds)
    threshold_tensor = torch.tensor(
        [label["lower_bound"] for label in thresholds.values()]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return feature_tensor > threshold_tensor


def create_intervals(
    thresholded_features: torch.Tensor,
    conv_settings: ConvolutionSettings,
    label_encoder: MultiLabelEncoder,
) -> Intervals:
    """Given the frames and their activated label, construct the intervals.

    Args:
        thresholded_features (torch.Tensor): Tensor of 0 and 1 values, compatible with multi-label classification.
        conv_settings (ConvolutionSettings): settings used to get the frame receptive fields boundaries
        label_encoder (MultiLabelEncoder): Label encoder used containing the labels

    Returns:
        Intervals: Structure containing the final intervals.
    """
    intervals = Intervals()

    # ( positive_instance, 2: (feat_i, label_i) )
    indices = torch.argwhere(thresholded_features)
    for feat_i, label_i in indices:
        feat_i_item = int(feat_i.item())
        fs = conv_settings.rf_start_i(feat_i_item)

        frame_start = max(0, fs)
        frame_end = fs + conv_settings.rf_step  # .rf_end_i(feat_i_item) + 1
        label = label_encoder.inv_transform(int(label_i.item()))

        intervals.add((frame_start, frame_end, label))
    return intervals


def infer(
    audio_path: Path,
    model: BaseSegmentationModel,
    output_p: Path,
    config: Config,
    batch_size: int,
    thresholds: None | dict = None,
):
    """Apply the model on the audio in a streaming fashion to ensure memory integrity,
    threshold the features, retrieve the intervals and write them to disk.

    Args:
        audio_path (Path): Path to the audio to process.
        model (BaseSegmentationModel): model used to perform inference.
        output_p (Path): output Path that will contain the detections.
        config (Config): config file of the model.
        batch_size (int): batch size to use for the forward pass.
        thresholds (None | dict, optional): threshold dict to use. Defaults to None.
    """
    if thresholds is None:
        thresholds = {
            label: {
                "lower_bound": 0.5,
                "upper_bound": 1.0,
            }
            for label in model.label_encoder._labels
        }
    inference_settings = ConvolutionSettings(
        kernels=(320,),
        strides=(320,),
        paddings=(0,),
    )

    # NOTE - apply model on audio
    logits_t = apply_model_on_audio(
        audio_path=audio_path,
        model=model,
        batch_size=batch_size,
        chunk_duration_s=config.audio.chunk_duration_s,
    )

    # NOTE - apply tresholding
    thresholded_features = (
        apply_thresholds(logits_t, thresholds=thresholds).detach().cpu()
    )

    # NOTE - cerate intervals
    intervals = create_intervals(
        thresholded_features=thresholded_features,
        conv_settings=inference_settings,
        label_encoder=model.label_encoder,
    )

    # NOTE - write intervals to disk
    write_intervals(intervals=intervals, audio_path=audio_path, output_p=output_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for inference.",
    )
    parser.add_argument("--uris", help="list of uris to use for prediction")
    parser.add_argument("--wavs", default="data/debug/wav")
    parser.add_argument(
        "--checkpoint",
        default="models/last/best.ckpt",
        help="Path to a pretrained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--thresholds",
        help="Path to a threshold dict, perform predictions using these threshols, otherwise use default value of .5.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Size of the batch used for the forward pass in the model.",
    )

    args = parser.parse_args()
    args.wavs = Path(args.wavs)
    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)

    if not args.wavs.exists():
        raise ValueError(f"Path `{args.wavs=}` does not exists")
    if not args.checkpoint.exists():
        raise ValueError(f"Path `{args.checkpoint=}` does not exists")
    if args.thresholds:
        if not Path(args.thresholds).exists():
            raise ValueError("Path to a valid threshold dict does not exist.")
        with Path(args.thresholds).open("r") as f:
            args.thresholds = yaml.safe_load(f)

    config: Config = load_config(args.config)

    if "hydra" not in config.model.name:
        raise ValueError("only MLE for the moment")
    l_encoder = MultiLabelEncoder(labels=config.data.classes)

    model = Models[config.model.name].load_from_checkpoint(
        checkpoint_path=args.checkpoint, label_encoder=l_encoder, config=config
    )
    model.eval()

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # model = torch.compile(model)

    # NOTE - get files and run inference
    if args.uris:
        with Path(args.uris).open("r") as uri_f:
            files_to_infer_on = [
                (args.wavs / uri.strip()).with_suffix(".wav")
                for uri in uri_f.readlines()
            ]
    else:
        files_to_infer_on = sorted(list(args.wavs.glob("*.wav")))
    n_files = len(files_to_infer_on)

    for i, audio_path in enumerate(sorted(files_to_infer_on), 1):
        print(
            f"[log] - ({i:>{len(str(n_files))}}/{n_files}) - running inference for file: '{audio_path.stem}'",
            flush=True,
        )

        infer(
            audio_path=audio_path,
            model=model,
            output_p=args.output,
            config=config,
            thresholds=args.thresholds,
            batch_size=args.batch_size,
        )
