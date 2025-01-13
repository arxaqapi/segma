from math import ceil
from pathlib import Path

import torch
import torchaudio

from segma.annotation import AudioAnnotation
from segma.models.base import BaseSegmentationModel
from segma.structs.interval import Interval, Intervals
from segma.utils.conversions import frames_to_milliseconds
from segma.utils.receptive_fields import rf_end_i, rf_start_i


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
    rttm_out = output_p / "rttm"
    aa_out = output_p / "aa"
    rttm_out.mkdir(exist_ok=True, parents=True)
    aa_out.mkdir(exist_ok=True, parents=True)

    uri = audio_path.stem
    with (
        (rttm_out / uri).with_suffix(".rttm").open("w") as rttm_f,
        (aa_out / uri).with_suffix(".aa").open("w") as aa_f,
    ):
        for start_f, end_f, label in all_intervals:
            aa = AudioAnnotation(
                uid=audio_path.stem,
                start_time_ms=float(frames_to_milliseconds(start_f)),
                duration_ms=float(frames_to_milliseconds(end_f - start_f)),
                label=str(label),
            )
            rttm_f.write(aa.to_rttm() + "\n")
            aa_f.write(aa.write() + "\n")
