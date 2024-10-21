from math import ceil, floor
from pathlib import Path

import torch
import torchaudio

from segma.annotation import AudioAnnotation
from segma.models import BaseSegmentationModel, Whisperidou
from segma.structs.interval import Interval, Intervals
from segma.utils.conversions import frames_to_milliseconds
from segma.utils.receptive_fields import rf_end_i, rf_start_i


def prediction(
    audio_path: Path, model: BaseSegmentationModel, output_p: Path = Path("out")
):
    """takes a path to an audio file and a trained model,
    perform frame-level inference and stitch everything together
    """
    if not output_p.exists():
        output_p.mkdir(exist_ok=True)
    assert audio_path.exists()

    audio_t, _sr = torchaudio.load(audio_path)
    assert _sr == 16_000

    # downmix to mono
    if audio_t.shape[0] > 1:
        audio_t = audio_t.mean(dim=0)
    audio_t = audio_t.squeeze()

    # create chunks of size chunk_size by padding and reshaping the audio
    n_frames = len(audio_t)
    chunck_size_f = 32_000

    # if < 1 pad pad to match batch, if > 1 pad an extra batch
    max_number_of_chunks = n_frames / chunck_size_f
    missing_n_frames = ceil(max_number_of_chunks) * chunck_size_f - n_frames
    if missing_n_frames > 0:
        audio_t = torch.nn.functional.pad(
            audio_t, pad=(0, missing_n_frames), mode="constant", value=0
        )

    # (batch, 32_000)
    batch_t = audio_t.reshape((ceil(max_number_of_chunks), chunck_size_f))

    # NOTE - pass batch_t through model
    # predicted output, to be processed
    # whisperidou specific
    if isinstance(model, Whisperidou):
        batch_t = model.audio_preparation_hook(batch_t.cpu().numpy())["input_features"]

    # (batch, windows, n_labels)
    output = model(batch_t)
    print(f"{output.shape=}")

    print(model.label_encoder.rev_map)
    all_intervals = Intervals()
    for batch_i, batch in enumerate(output):
        # Zip with prediction windows. or get them through computation (clip)

        # NOTE - for each window, create an interval per label
        # merge intervals by adding them to the Intervals structure
        for w_i, window in enumerate(batch):
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
    #     if batch_i == 2:
    #         print(all_intervals)
    #         exit(55)
    # TODO remove x last prediction if audio was padded
    if missing_n_frames > 0:
        raise NotImplementedError

    # TODO generate aa & rttms
    for start_f, end_f, label in all_intervals:
        aa = AudioAnnotation(
            uid=audio_path.stem,
            start_time_ms=float(frames_to_milliseconds(start_f)),
            duration_ms=float(frames_to_milliseconds(end_f - start_f)),
            label=str(label),
        )

        print(aa.to_rttm())
