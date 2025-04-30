from functools import partial
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import torch
import torchaudio

from segma.config.base import Config
from segma.models.base import BaseSegmentationModel, ConvolutionSettings
from segma.predict import write_intervals
from segma.structs.interval import Intervals
from segma.utils.encoders import MultiLabelEncoder


def next_batch_start_i(chunk_duration_f: int, cs: ConvolutionSettings) -> int:
    """Return the input range (receprive field span) covered by the model output features.

    This is used to perform perfectly calibrated sliding window prediction without loosing a single frame.

    ```
    |------------------|    <- audio
      ..... |----|
                |----|      <- last valid batch
                    |----|  <- batch in next chunk
    ```

    Args:
        chunk_duration_f (int): Size of the audio chunks to consider that will be part of the batch.
        cs (ConvolutionSettings): Receptive Field settings of the model

    Returns:
        int: start index of the next batch.
    """
    last_covered_chunk_s_i = cs.n_windows(chunk_duration_f, True) * cs.rf_step()
    next_batch_first_chunk_i = last_covered_chunk_s_i + cs.rf_step() - cs.rf_start_i(0)

    # NOTE - last chunk i-s
    assert last_covered_chunk_s_i < chunk_duration_f
    assert last_covered_chunk_s_i + cs.rf_size() < chunk_duration_f
    # NOTE - next chunk i-s
    assert last_covered_chunk_s_i + cs.rf_step() < chunk_duration_f
    assert last_covered_chunk_s_i + cs.rf_step() + cs.rf_size() > chunk_duration_f

    return next_batch_first_chunk_i


def audio_slicer(start: int, n_frames: int, audio_path: Path) -> torch.Tensor:
    """Slices an audio file given an path `audio_path` for `n_frames`
    starting from frame `start`.

    If the start value is negative, the audio is padded for abs(start) frames at the
    beginning with 0 values.

    Args:
        audio_path (Path): Path to the audio file to load.
        start (int): Start frame to start loading.
        n_frames (int): Number of frames to load.

    Returns:
        torch.Tensor: Loaded audio waveform.
    """
    to_pad = 0
    if start < 0:
        to_pad = abs(start)
        start = 0
        n_frames -= to_pad

    audio_t = torchaudio.load(
        uri=audio_path.resolve(),
        frame_offset=start,
        num_frames=n_frames,
    )[0]
    audio_t = torch.nn.functional.pad(audio_t, pad=(to_pad, 0), mode="constant")
    return audio_t


def generic_batched_windows(
    object_length: int,
    loader: Callable[[int, int], torch.Tensor],
    step: int,
    size: int,
    *,
    n: int = 8,
    start: int = 0,
) -> Generator[torch.Tensor, None, None]:
    """Given an audio, return a generator that performs batched-sliding windows.

    The function returs a batch of size `n`. Where each element of the batch is a window (of size `size`)
    with displacement rules defined by the `start`, `step` and `size` parameters.

    The function is lazy be default since it returns a generator, such that not the complete audio is loaded
    all at once but each batch can be processed before querying the next one.

    Args:
        object_length (int): Length of the object to slide over.
        loader (Callable[[int, int], torch.Tensor]): function that takes a start index and a size argument and returns the slice of the object.
        step (int): Step size of the sliding window. If `step < size` then the windows will overlap.
        size (int): Size of the window to take into account.
        n (int, optional): Batch size. Defaults to 8.
        start (int, optional): Start position of the first window. Defaults to 0.

    Raises:
        NotImplementedError: Raised if `start<0`. To be implemented for Whisper

    Yields:
        Generator[torch.Tensor, None, None]: Returns a generator that yields tensors stacked on the first dimension.
    """
    chunk_starts = list(range(start, object_length - size + 1, step))
    n_chunks = len(chunk_starts)
    missing = object_length - n_chunks * step

    i = 0
    while i < n_chunks:
        batch = []
        for _ in range(min(n, n_chunks - i)):
            start_i = chunk_starts[i]
            batch.append(loader(start_i, size))  # eq. O[s_i : s_i + size]
            i += 1
        yield torch.stack(batch, dim=0)

    # NOTE - handle missing elements
    if missing:
        last_incomplete_batch_i = n_chunks * step
        # eq. O[s_i: ]
        yield torch.stack([loader(last_incomplete_batch_i, -1)], dim=0)


def apply_model_on_audio(
    audio_path: Path,
    model: BaseSegmentationModel,
    chunk_duration_f: int,
    batch_size: int = 8,
) -> torch.Tensor:
    if not isinstance(model.label_encoder, MultiLabelEncoder):
        raise ValueError("Only hydra models supported for the sake of simplicity")

    batches = generic_batched_windows(
        object_length=torchaudio.info(audio_path).num_frames,
        loader=partial(audio_slicer, audio_path=audio_path),
        start=model.conv_settings.rf_start_i(0),
        step=next_batch_start_i(chunk_duration_f, model.conv_settings),
        size=chunk_duration_f,
        n=batch_size,
    )

    outputs = []
    for batch in batches:
        batch = model.audio_preparation_hook(batch.squeeze().cpu().numpy())
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(
            torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        )
        with torch.inference_mode():
            # in : (batch, 80, 3000)
            # out: (batch, n_feat, 1, n_labels)
            outputs.append(model(batch).squeeze(2).flatten(0, 1))
    return torch.concat(outputs)


def apply_tresholds(
    feature_tensor: torch.Tensor, tresholds: dict[str, dict[str, float]]
) -> torch.Tensor:
    feature_tensor = feature_tensor.sigmoid()
    assert feature_tensor.shape[-1] == len(tresholds)
    treshold_tensor = torch.tensor(
        [label["lower_bound"] for label in tresholds.values()]
    ).to(torch.device("mps" if torch.backends.mps.is_available() else "cuda"))
    active = feature_tensor > treshold_tensor

    return active


def create_intervals(
    tresholded_features: torch.Tensor,
    conv_settings: ConvolutionSettings,
    label_encoder: MultiLabelEncoder,
) -> Intervals:
    intervals = Intervals()

    # ( positive_instance, 2: (feat_i, label_i) )
    indices = torch.argwhere(tresholded_features)
    for feat_i, label_i in indices:
        # REVIEW - cache results in rf_comp
        frame_start = max(0, conv_settings.rf_start_i(int(feat_i.item())))
        frame_end = conv_settings.rf_end_i(int(feat_i.item())) + 1
        label = label_encoder.inv_transform(int(label_i.item()))

        intervals.add((frame_start, frame_end, label))
    return intervals


def predict(
    audio_path: Path,
    model: BaseSegmentationModel,
    output_p: Path,
    config: Config,
    thresholds: dict[str, dict[str, float]] | None,
    batch_size: int = 128,
) -> None:
    # TODO - min_duration_on / off to implement (filter Intervals list) (100ms)

    if thresholds is None:
        thresholds = {
            label: {
                "lower_bound": 0.5,
                "upper_bound": 1.0,
            }
            for label in model.label_encoder._labels
        }

    print("[log | debug] - apply_model_on_audio")
    feature_tensor = apply_model_on_audio(
        audio_path=audio_path,
        model=model,
        chunk_duration_f=config.audio.chunk_duration_f,
        batch_size=batch_size,
    )

    print("[log | debug] - apply_tresholds")
    tresholded_features = apply_tresholds(feature_tensor, tresholds=thresholds)

    print("[log | debug] - create_intervals")
    intervals = create_intervals(
        tresholded_features=tresholded_features,
        conv_settings=model.conv_settings,
        label_encoder=model.label_encoder,
    )

    print("[log | debug] - write_intervals")
    write_intervals(intervals=intervals, audio_path=audio_path, output_p=output_p)


if __name__ == "__main__":
    import argparse

    import yaml

    from segma.config import load_config
    from segma.models import Models

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
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
    parser.add_argument(
        "--thresholds",
        help="If thresholds dict is given, perform predictions using thresholding.",
    )

    args = parser.parse_args()
    args.wavs = Path(args.wavs)
    args.ckpt = Path(args.ckpt)
    if args.thresholds is not None and Path(args.thresholds).exists():
        with Path(args.thresholds).open("r") as f:
            threshold_dict = yaml.safe_load(f)
    else:
        threshold_dict = None

    if not args.wavs.exists():
        raise ValueError(f"Path `{args.wavs=}` does not exists")
    if not args.ckpt.exists():
        raise ValueError(f"Path `{args.ckpt=}` does not exists")

    cfg: Config = load_config(args.config)

    if "hydra" in cfg.model.name:
        l_encoder = MultiLabelEncoder(labels=cfg.data.classes)
    else:
        raise ValueError("only MLE for the moment")

    # NOTE - resolve output_path
    # if path is model/last/best -> resolve symlink
    if args.output is None and str(args.ckpt) == "models/last/best.ckpt":
        args.output = Path("/".join(args.ckpt.resolve().parts[-4:-2]))
    elif args.output is None:
        try:
            args.output = Path("/".join(args.ckpt.parts[-4:-2]))
        except:
            args.output = Path("segma_out")
    else:
        args.output = Path(args.output)

    model = Models[cfg.model.name].load_from_checkpoint(
        checkpoint_path=args.ckpt, label_encoder=l_encoder, config=cfg
    )

    model.to(torch.device("mps" if torch.backends.mps.is_available() else "cuda"))
    if cfg.model.name in ("hydra_whisper", "HydraWhisper"):
        torch._dynamo.config.accumulated_cache_size_limit = 32
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 32
        model = torch.compile(model)

    # NOTE if args.uris: path is known
    if args.uris:
        with Path(args.uris).open("r") as uri_f:
            uris = [uri.strip() for uri in uri_f.readlines()]
        for uri in uris:
            wav_f = (args.wavs / uri).with_suffix(".wav")
            print(f"[log] - running inference for file: '{wav_f.stem}'")
            predict(
                audio_path=wav_f,
                model=model,
                output_p=args.output,
                config=cfg,
                thresholds=threshold_dict,
            )
    else:
        # TODO - implement folder based prediction
        raise ValueError("uris should be provided")
