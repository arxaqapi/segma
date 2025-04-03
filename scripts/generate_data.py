from pathlib import Path

import numpy as np
from scipy.io import wavfile

from segma.annotation import AudioAnnotation


def gen_annots(
    uid: str,
    audio_duration_s: float = 60.0,
    labels: list[str] = ["male", "female", "key_child", "other_child"],
    max_annot_duration_s: int = 3,
    min_annot_count: int = 4,
    max_annot_count: int = 10,
) -> list[AudioAnnotation]:
    rng = np.random.default_rng()

    # get n between 4 to 10
    n = np.random.randint(min_annot_count, max_annot_count)
    # get n random duration
    durations_s = rng.uniform(0.2, max_annot_duration_s, size=n)
    # get n starting points
    starting_points_s = rng.uniform(0, audio_duration_s - max_annot_duration_s, size=n)
    # NOTE - second_to_millisecond
    # durations = second_to_millisecond(durations_s)
    # starting_points: np.ndarray = second_to_millisecond(starting_points_s)

    starting_points_s.sort()
    label_idxs = rng.integers(len(labels), size=n)

    annotations = [
        AudioAnnotation(
            uid=uid, start_time_s=start_s, duration_s=dur_s, label=labels[label_i]
        )
        for start_s, dur_s, label_i in zip(starting_points_s, durations_s, label_idxs)
    ]
    return annotations


def gen_audio_from_annot(
    annotations: list[AudioAnnotation],
    label_to_freq: dict[str, int],
    audio_duration_s: float = 60.0,
    sample_rate: int = 16_000,
):
    # TODO - normalize audio
    n_samples = int(audio_duration_s * sample_rate)
    array = np.zeros((1, n_samples), dtype=np.float32)
    for annot in annotations:
        start_time_f = int(annot.start_time_s * sample_rate)
        duration_f = int(annot.duration_s * sample_rate)

        freq_segment = gen_sine(
            f=label_to_freq[annot.label], duration_s=annot.duration_s
        )

        array[:, start_time_f : start_time_f + duration_f] = freq_segment
    return array


def gen_sine(f: int = 440, duration_s: float = 1.0, sr: int = 16_000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sine_wave = np.sin(2 * np.pi * f * t)
    return sine_wave


def _plot_spectro(waveform, fig_title: str = "audio.png", sr: int = 16_000):
    """Generate a spectorgram of the given waveform and writes it to disk as a `.png` image."""
    import matplotlib.pyplot as plt
    import scipy as sp

    f, t, zz = sp.signal.stft(waveform, fs=sr)

    fig, ax = plt.subplots(dpi=150)
    # spectro are optional, remove expensive gouraud
    pcm = ax.pcolormesh(t, f, np.abs(zz))  # , shading="gouraud")

    ax.set_ylim(top=4000)

    plt.title("Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(pcm, label="Magnitude")
    plt.savefig(fig_title)
    plt.close(fig)


def gen_classification(
    output: Path = Path("data"),
    audio_duration_s: float = 60.0,
    labels: list[str] = ["male", "female", "key_child", "other_child"],
    per_split: int = 5,
    gen_spectro: bool = False,
):
    wav_out = output / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    annotations_out = output / "aa"
    annotations_out.mkdir(parents=True, exist_ok=True)

    rttms_out = output / "rttm"
    rttms_out.mkdir(parents=True, exist_ok=True)

    uems_out = output / "uem"
    uems_out.mkdir(parents=True, exist_ok=True)

    spectro_out = output / "spectrograms"
    spectro_out.mkdir(parents=True, exist_ok=True)
    # NOTE - 1. generate uid list
    _uids = [str(i).rjust(4, "0") for i in range(3 * per_split)]
    uids = {
        split: _uids[i * per_split : (i + 1) * per_split]
        for i, split in enumerate(("train", "val", "test"))
    }

    # NOTE - write uid list
    for split, s_uids in uids.items():
        with (output / f"{split}.txt").open("w") as f:
            for uid in s_uids:
                f.write(uid + "\n")

    label_to_freq = {label: 440 * i for i, label in enumerate(labels, start=1)}

    for split, s_uids in uids.items():
        for uid in s_uids:
            # NOTE - gen annotations
            annots = gen_annots(
                uid,
                audio_duration_s=audio_duration_s,
                labels=labels,
                # min_annot_count=15,
                # max_annot_count=25,
            )
            # NOTE - map annotations to frequency and generate audio
            # (channels, samples)
            audio = gen_audio_from_annot(
                annots, label_to_freq, audio_duration_s=audio_duration_s
            )
            if gen_spectro:
                _plot_spectro(
                    audio[0], fig_title=str(spectro_out / f"{uid}_spectro.png")
                )

            # NOTE - write wav file and annotations (.aa, .rttm, .uem)
            wavfile.write((wav_out / uid).with_suffix(".wav"), 16_000, audio.T)

            with (annotations_out / f"{uid}.aa").open("w") as f:
                f.writelines([a.write() + "\n" for a in annots])

            with (rttms_out / f"{uid}.rttm").open("w") as f:
                f.writelines([a.to_rttm() + "\n" for a in annots])
            # corresponding UEM > fixed audio duration
            with (uems_out / f"{uid}.uem").open("w") as f:
                f.write(f"{uid} NA 0.000 {audio_duration_s}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        required=True,
        help="Number of examples per split",
        type=int,
    )
    parser.add_argument(
        "--spectrogram",
        help="Boolean flag to determine if spectograms need to be generated.",
        action="store_true",
    )
    args = parser.parse_args()

    db_path = Path(f"data/debug_{args.n_samples}")
    if not db_path.exists():
        print(
            f"[log] - Generating a dummy dataset of size {args.n_samples * 3} ({args.n_samples} * 3)."
        )
        gen_classification(
            output=db_path, per_split=args.n_samples, gen_spectro=args.spectrogram
        )
    else:
        print("[log] - dataset already exists, nothing will happen.")
