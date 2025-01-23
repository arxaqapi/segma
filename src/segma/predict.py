from math import ceil
from pathlib import Path

import numpy as np
import torch
import torchaudio

from segma.annotation import AudioAnnotation
from segma.config.base import Config
from segma.models.base import BaseSegmentationModel
from segma.structs.interval import Interval, Intervals
from segma.utils.conversions import frames_to_milliseconds
from segma.utils.receptive_fields import rf_end_i, rf_start_i


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
                start_time_ms=float(frames_to_milliseconds(start_f)),
                duration_ms=float(frames_to_milliseconds(end_f - start_f)),
                label=str(label),
            )
            rttm_f.write(aa.to_rttm() + "\n")
            aa_f.write(aa.write() + "\n")


def prediction(audio_path: Path, model: BaseSegmentationModel, output_p: Path):
    """Takes a path to an audio file, a trained model and output folder,
    perform frame-level inference, stitches everything together
    and saves `.aa`and `.rttm` files to the output folder and their corresponding subfolders.

    Args:
        audio_path (Path): Path to a collection of `.wav` files.
        model (BaseSegmentationModel): Trained model used to perform inference.
        output_p (Path): Output folder that will contain the segmentation files.
    """
    assert audio_path.exists()

    audio_t, _sr = torchaudio.load(audio_path.resolve())
    assert _sr == 16_000

    # downmix to mono
    if audio_t.shape[0] > 1:
        audio_t = audio_t.mean(dim=0)
    audio_t = audio_t.squeeze()

    # create chunks of size chunk_size by padding and reshaping the audio
    n_valid_frames = len(audio_t)
    chunck_size_f = 2 * 16_000  # _sr

    # if < 1 pad pad to match batch, if > 1 pad an extra batch
    max_number_of_chunks = n_valid_frames / chunck_size_f
    missing_n_frames = ceil(max_number_of_chunks) * chunck_size_f - n_valid_frames
    if missing_n_frames > 0:
        audio_t = torch.nn.functional.pad(
            audio_t, pad=(0, missing_n_frames), mode="constant", value=0
        )
    # (batch, 32_000) | batch_t[-1][-missing_n_frames] = first padded value
    batch_t = audio_t.reshape((ceil(max_number_of_chunks), chunck_size_f))

    # NOTE - pass batch_t through model
    # predicted output, to be processed
    batch_t = model.audio_preparation_hook(batch_t.cpu().numpy())
    batch_t = torch.tensor(batch_t)

    # (batch, windows, n_labels)
    model.eval()
    with torch.no_grad():
        output = model(batch_t)

    all_intervals = Intervals()
    for batch_i, batch in enumerate(output):
        # batch: (windows, n_labels)
        # Zip with prediction windows. or get them through computation (clip)

        # NOTE - for each window, create an interval per label
        # merge intervals by adding them to the Intervals structure
        for w_i, window in enumerate(batch):
            # windows: (n_labels, )
            frame_start = batch_i * chunck_size_f + rf_start_i(
                w_i, model.conv_settings.strides, model.conv_settings.paddings
            )
            frame_end = batch_i * chunck_size_f + rf_end_i(
                w_i,
                model.conv_settings.kernels,
                model.conv_settings.strides,
                model.conv_settings.paddings,
            )
            found_labels = model.label_encoder.inv_transform(window.argmax().item())

            for label in found_labels:
                corresponding_interval: Interval = (frame_start, frame_end, label)
                all_intervals.add(corresponding_interval)

    # TODO remove x last prediction if audio was padded
    if missing_n_frames > 0:
        for interval in all_intervals:
            # remove from interlap / create new interlap object
            if interval[0] > n_valid_frames - missing_n_frames:
                pass
                # raise NotImplementedError

    # NOTE - generate & write; aa & rttms
    write_intervals(intervals=all_intervals, audio_path=audio_path, output_p=output_p)


# TODO - add chunck_size_f as parameter of function
def sliding_prediction(
    audio_path: Path, model: BaseSegmentationModel, output_p: Path, config: Config
):
    """do not open audio entirely
    - perform slide-wise"""
    batch_size = 32
    chunck_size_f = (
        int(config.audio_config.chunk_duration_s) * config.audio_config.sample_rate
    )
    meta_b_size = batch_size * chunck_size_f

    audio_info = torchaudio.info(audio_path.resolve())
    number_frames = audio_info.num_frames

    max_meta_batches = ceil(number_frames / meta_b_size)
    # NOTE - for each meta_batch pass 32 batches through the model

    reference_windows = gen_bounds(max_value=chunck_size_f)[
        : model.conv_settings.n_windows(
            chunk_duration_f=config.audio_config.chunk_duration_f,
            strict=config.audio_config.strict_frames,
        )
    ]
    # REVIEW - n_frames is too restrictive, this hackeridou is awfull
    reference_windows[-1][-1] = chunck_size_f

    all_intervals = Intervals()

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
        # REVIEW
        batch_t = torch.clone(batch_t)

        # NOTE - pass batch through model
        model.eval()
        with torch.no_grad():
            output_t = model(batch_t) # .to(torch.device("mps")))

        # NOTE - using windows of size 20ms in model output
        for batch_i, batch in enumerate(output_t):
            # batch: (windows, n_labels)
            predictions = batch.argmax(-1)
            for pred, (w_start, w_end) in zip(predictions, reference_windows):
                offset = start_i + batch_i * chunck_size_f
                frame_start = w_start + offset
                # FIXME - Intervals needs to be fixed
                frame_end = w_end + offset + 1
                # NOTE - skip padded section
                if n_frames_to_pad > 0 and frame_start > (end_i - n_frames_to_pad):
                    break
                    # print(
                    #     "[log] - we stop the recording of predictions since we are in padded zone now"
                    # )

                found_labels = model.label_encoder.inv_transform(pred.item())
                for label in found_labels:
                    corresponding_interval: Interval = (frame_start, frame_end, label)
                    all_intervals.add(corresponding_interval)

    # NOTE - generate & write; aa & rttms
    write_intervals(intervals=all_intervals, audio_path=audio_path, output_p=output_p)


def gen_bounds(
    window_size: int = 320,
    start_value: int = -160,
    max_value: int = 32_000,
    clip_values: tuple[int, int] = (0, 32_000),
) -> list[list[int]]:
    """generate list of bounds without overlap"""
    assert max_value == clip_values[1]

    return [
        [int(a), int(b)]
        for a, b in zip(
            np.arange(start_value, max_value + window_size, window_size).clip(
                *clip_values
            ),
            np.arange(window_size // 2, max_value + window_size, window_size).clip(
                *clip_values
            ),
        )
    ]

