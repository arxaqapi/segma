from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torchaudio

from segma.annotation import AudioAnnotation
from segma.config.base import Config
from segma.models.base import BaseSegmentationModel
from segma.structs.interval import Interval, Intervals
from segma.utils.conversions import frames_to_seconds
from segma.utils.encoders import (
    LabelEncoder,
    MultiLabelEncoder,
    PowersetMultiLabelEncoder,
)


def write_logits_to_disk(logits: np.ndarray, logits_p: Path | str) -> None:
    """Save the logits to disk as a `.npy` file.

    Args:
        logits (np.ndarray): Logits numpy array to be saved
        logits_p (Path | str): Output file path (without extension).
    """
    logits_p = Path(logits_p)
    logits_p.parent.mkdir(parents=True, exist_ok=True)

    with logits_p.with_suffix(".npy").open("wb") as bf:
        np.save(bf, logits)


def load_logit_from_disk(logits_p: Path | str) -> np.ndarray:
    """Load a single numpy array of logits from disk.

    Args:
        logits_p (Path | str): Path to the `.npy` file.

    Returns:
        np.ndarray: Loaded logits array.
    """
    logits_p = Path(logits_p)
    with logits_p.open("rb") as bf:
        logits_a = np.load(bf)
    return logits_a


def load_all_logits(logits_p: Path | str) -> dict[str, np.ndarray]:
    logits_p = Path(logits_p)
    assert logits_p.exists() and logits_p.is_dir()
    # .npy
    return {logit.stem: load_logit_from_disk(logit) for logit in logits_p.glob("*")}


def predict_all_logits(
    logits: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, float]],
    label_encoder: LabelEncoder,
) -> dict[str, list[AudioAnnotation]]:
    """Given a dict `logits` that maps uris to logits, perform prediction for each loaded logit
    and return a dict that maps uris to predictions.

    Args:
        logits (dict[str, np.ndarray]): _description_
        thresholds (dict[str, dict[str, float]]): Treshold dict to take into account when predicting from logits.
        label_encoder (LabelEncoder): _description_

    Returns:
        dict[str, list[AudioAnnotation]]: A dict that maps an uri to the predicted annotations.
    """
    return {
        uri: predict_from_logits_with_thresholds(logit, uri, thresholds, label_encoder)
        for uri, logit in logits.items()
    }


def predict_from_logits_with_thresholds(
    logits: np.ndarray,
    uri: str,
    thresholds: dict[str, dict[str, float]],
    label_encoder: LabelEncoder,
) -> list[AudioAnnotation]:
    """Given a logit array, its corresponding uri and the thresholds to use, perform

    Args:
        logits (np.ndarray): _description_
        uri (str): _description_
        thresholds (dict[str, dict[str, float]]): _description_
        label_encoder (LabelEncoder): _description_

    Returns:
        list[AudioAnnotation]: _description_
    """
    assert isinstance(label_encoder, MultiLabelEncoder)

    all_intervals = Intervals()

    for start_f, end_f, logit in logits:
        # logit: (4,)
        for label_logit, label in zip(logit, label_encoder.base_labels):
            if (
                thresholds[label]["lower_bound"]
                <= label_logit
                <= thresholds[label]["upper_bound"]
            ):
                corresponding_interval: Interval = (
                    start_f,
                    end_f,
                    label,
                )
                all_intervals.add(corresponding_interval)

    return interval_to_aa(all_intervals, uri)


def interval_to_aa(intervals: Intervals, uri: str) -> list[AudioAnnotation]:
    """Transforms an `Intervals` object into a list of `AudioAnnotation`s."""
    return [
        AudioAnnotation(
            uid=uri,
            start_time_s=float(frames_to_seconds(start_f)),
            duration_s=float(frames_to_seconds(end_f - start_f)),
            label=str(label),
        )
        for start_f, end_f, label in intervals
    ]


def write_intervals(intervals: Intervals, audio_path: Path, output_p: Path):
    rttm_out = output_p / "rttm"
    aa_out = output_p / "aa"
    rttm_out.mkdir(exist_ok=True, parents=True)
    aa_out.mkdir(exist_ok=True, parents=True)

    uri = audio_path.stem
    with (
        (rttm_out / uri).with_suffix(".rttm").open("w") as rttm_f,
        (aa_out / uri).with_suffix(".aa").open("w") as aa_f,
    ):
        for start_f, end_f, label in intervals:
            aa = AudioAnnotation(
                uid=audio_path.stem,
                start_time_s=float(frames_to_seconds(start_f)),
                duration_s=float(frames_to_seconds(end_f - start_f)),
                label=str(label),
            )
            rttm_f.write(aa.to_rttm() + "\n")
            aa_f.write(aa.write() + "\n")


# TODO - add chunck_size_f as parameter of function
def sliding_prediction(
    audio_path: Path,
    model: BaseSegmentationModel,
    output_p: Path,
    config: Config,
    save_logits: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
    batch_size: int = 128,
):
    """do not open audio entirely
    - perform slide-wise"""
    model.eval()
    chunck_size_f = int(config.audio.chunk_duration_s) * config.audio.sample_rate
    meta_b_size = batch_size * chunck_size_f

    audio_info = torchaudio.info(audio_path.resolve())
    number_frames = audio_info.num_frames

    max_meta_batches = ceil(number_frames / meta_b_size)
    # NOTE - for each meta_batch pass `batch_size` batches through the model
    reference_windows = gen_bounds(
        max_value=chunck_size_f, clip_values=(0, chunck_size_f)
    )[
        : model.conv_settings.n_windows(
            chunk_duration_f=config.audio.chunk_duration_f,
            strict=config.audio.strict_frames,
        )
    ]
    # REVIEW - n_frames is too restrictive, this hackeridou is awfull (31xxx -> 32000)
    reference_windows[-1][-1] = chunck_size_f

    all_intervals = Intervals()

    if save_logits:
        # NOTE - define logits datatype
        mem_logit_dt = np.dtype(
            [
                ("start_f", np.int32),
                ("end_f", np.int32),
                ("predictions", np.float32, (len(model.label_encoder.base_labels),)),
            ]
        )
        # list of numpy array, one line per frame prediction ()
        raw_logits: dict[tuple[int, int], list[float]] = defaultdict(list)

    for meta_batch_i in range(max_meta_batches):
        start_i = meta_batch_i * meta_b_size
        end_i = min(number_frames, (meta_batch_i + 1) * meta_b_size)

        max_number_batches = ceil((end_i - start_i) / chunck_size_f)

        # NOTE - load only necessary portion of the audio
        audio_t, _sr = torchaudio.load(
            audio_path.resolve(), frame_offset=start_i, num_frames=end_i - start_i
        )
        # NOTE - if last meta_batch, look if padding necessary
        # TODO - improve padding mechanism
        expected_n_frames = max_number_batches * chunck_size_f
        n_frames_to_pad = expected_n_frames - (end_i - start_i)
        if meta_batch_i == (max_meta_batches - 1) and n_frames_to_pad > 0:
            audio_t = torch.nn.functional.pad(
                audio_t, pad=(0, n_frames_to_pad), mode="constant", value=0
            )
            end_i = number_frames + n_frames_to_pad
        # NOTE - reshape into a batch using `max_number_batches``

        batch_t = audio_t.reshape(max_number_batches, chunck_size_f)

        # NOTE - pass batch_t through model
        # predicted output, to be processed
        # (batch, windows, n_labels)
        batch_t = model.audio_preparation_hook(batch_t.cpu().numpy())
        # REVIEW need to check if it's appropriate
        #batch_t = torch.clone(batch_t)
        batch_t = torch.from_numpy(batch_t)

        # NOTE - pass batch through model
        with torch.no_grad():
            output_t: dict[str, torch.Tensor] | torch.Tensor = model(
                batch_t.to(
                    torch.device("mps" if torch.backends.mps.is_available() else "cuda")
                )
            )
            if "hydra" in config.model.name:
                for key, head_output_t in output_t.items():
                    output_t[key] = torch.nn.functional.sigmoid(head_output_t)

        # NOTE - only for MultiLabel models: x * (30, 99, 1)
        # TODO handle using label_encoder (thats its usecase)
        if isinstance(model.label_encoder, MultiLabelEncoder):
            # NOTE - one pass per label
            # mb_logits: dict[tuple[int, int], list[float]] = defaultdict(list)
            for key, head_output_t in output_t.items():
                head_label = key.removeprefix("linear_head_")
                for batch_i, batch in enumerate(head_output_t):
                    if thresholds is not None:
                        # NOTE - thresholds batch
                        label_prediction = (
                            (batch >= thresholds[head_label]["lower_bound"])
                            & (batch <= thresholds[head_label]["upper_bound"])
                        ).int()
                    else:
                        # NOTE - use base value of 0.5 for tresholding
                        label_prediction = (batch > 0.5).int()
                    for pred, raw_logit, (w_start, w_end) in zip(
                        label_prediction, batch, reference_windows
                    ):
                        offset = start_i + batch_i * chunck_size_f
                        frame_start = w_start + offset
                        # FIXME - Intervals needs to be fixed
                        frame_end = w_end + offset + 1
                        # NOTE - skip padded section
                        if n_frames_to_pad > 0 and frame_start > (
                            end_i - n_frames_to_pad
                        ):
                            break
                        found_labels = [] if pred == 0 else [head_label]
                        for label in found_labels:
                            corresponding_interval: Interval = (
                                frame_start,
                                frame_end,
                                label,
                            )
                            all_intervals.add(corresponding_interval)
                        # NOTE - handle logits saving
                        if save_logits:
                            raw_logits[(frame_start, frame_end)].append(
                                raw_logit.detach().cpu().numpy().item()
                            )

        elif isinstance(model.label_encoder, PowersetMultiLabelEncoder):
            # NOTE - using windows of size 20ms in model output
            for batch_i, batch in enumerate(output_t):
                # batch: (windows, n_labels)
                predictions = batch.argmax(-1)
                for pred, (w_start, w_end) in zip(predictions, reference_windows):
                    offset = start_i + batch_i * chunck_size_f
                    frame_start = w_start + offset
                    # FIXME - Intervals needs to be fixed instead of forcing the merge with a + 1
                    frame_end = w_end + offset + 1
                    # NOTE - skip padded section
                    if n_frames_to_pad > 0 and frame_start > (end_i - n_frames_to_pad):
                        break

                    found_labels = model.label_encoder.inv_transform(pred.item())
                    for label in found_labels:
                        corresponding_interval: Interval = (
                            frame_start,
                            frame_end,
                            label,
                        )
                        all_intervals.add(corresponding_interval)

    # NOTE - generate & write; aa & rttms
    write_intervals(intervals=all_intervals, audio_path=audio_path, output_p=output_p)

    # NOTE - save logits to disk
    if save_logits:
        raw_logits_array = np.array(
            [
                (start_f, end_f, np.array(logits))
                for (start_f, end_f), logits in raw_logits.items()
            ],
            dtype=mem_logit_dt,
        )
        print(f"[log] - saving logits to disk under '{output_p / 'logits'}'")
        write_logits_to_disk(
            logits=raw_logits_array, logits_p=output_p / "logits" / audio_path.stem
        )


def gen_bounds(
    window_size: int = 320,
    start_value: int = -160,
    max_value: int = 32_000,
    clip_values: tuple[int, int] = (0, 32_000),
) -> list[list[int]]:
    """Generate a list of non-overlapping interval bounds over a specified range.

    Each interval is defined by a start and end value, generated using a sliding window
    of a given size. The intervals are optionally clipped to remain within a specified range.

    Args:
        window_size (int, optional): Size of the window used to compute interval bounds. Defaults to 320.
        start_value (int, optional): Starting point for generating the first bound. Defaults to -160.
        max_value (int, optional): Upper limit for bound generation. Defaults to 32_000.
        clip_values (tuple[int, int], optional):
            Minimum and maximum values to clip the bounds to.
            Must match the range [min, max]. Defaults to (0, 32_000).

    Returns:
        list[list[int]]: A list of [start, end] interval bounds, clipped to the specified range.

    Note:
        Assumes that max_value is equal to clip_values[1]. Raises AssertionError otherwise.
    """
    assert max_value == clip_values[1]

    starts = np.arange(start_value, max_value + window_size, window_size)
    ends = np.arange(window_size // 2, max_value + window_size, window_size)

    return [
        [int(a), int(b)]
        for a, b in zip(
            starts.clip(*clip_values),
            ends.clip(*clip_values),
        )
    ]
