import argparse
import math
from pathlib import Path
from pprint import pprint

import sklearn
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from segma.config.base import load_config
from segma.data.utils import load_uris


def rttm_to_tensor(
    rttm_path: Path, labels: list[str], frame_resolution_s: float = 0.02
) -> torch.Tensor:
    """Convert RTTM file to multi-hot encoded tensor at specified resolution.

    Args:
        rttm_path (Path): Path to RTTM file.
        labels (list[str]): List of labels.
        frame_resolution_s (float, optional): Time resolution in seconds. Defaults to 0.02s (20ms).

    Returns:
        torch.Tensor: Of shape (num_frames, num_labels)
    """
    # Parse RTTM
    segments = []
    label_set = set(labels)
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            start_s = float(parts[3])
            duration_s = float(parts[4])
            label = parts[7]
            if label in label_set:
                segments.append((start_s, duration_s, label))

    # Build label mapping
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Determine number of frames
    total_duration = max(start + dur for start, dur, _ in segments) if segments else 0
    num_frames = math.ceil(total_duration / frame_resolution_s)

    # Create tensor and fill
    tensor = torch.zeros(num_frames, len(labels), dtype=torch.float32)

    for start, duration, label in segments:
        start_frame = int(start / frame_resolution_s)
        end_frame = min(math.ceil((start + duration) / frame_resolution_s), num_frames)
        tensor[start_frame:end_frame, label_to_idx[label]] = 1.0

    return tensor


def pad_tensors(tensor_0: torch.Tensor, tensor_1: torch.Tensor):
    """Given two tensors, pad the smallest one with zeroes to match the longest one on dim 0."""
    return (
        torch.nn.functional.pad(
            tensor_0, (0, 0, 0, max(0, tensor_1.shape[0] - tensor_0.shape[0]))
        ),
        torch.nn.functional.pad(
            tensor_1, (0, 0, 0, max(0, tensor_0.shape[0] - tensor_1.shape[0]))
        ),
    )


def unify(
    uri_to_logits_t_0: dict[str, torch.Tensor],
    uri_to_logits_t_1: dict[str, torch.Tensor],
    uris_to_load: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Makes sure the loaded tensor data is consistent shape-wise.
    Pad if not.
    """
    t0_map = {}
    t1_map = {}
    for uri in uris_to_load:
        t0, t1 = pad_tensors(
            uri_to_logits_t_0[uri],
            uri_to_logits_t_1[uri],
        )
        t0_map[uri] = t0
        t1_map[uri] = t1

    # stack loaded and padded tensors
    return torch.cat(list(t0_map.values()), dim=0), torch.cat(
        list(t1_map.values()), dim=0
    )


def load_pred_logits(
    logits_p: Path,
    labels: list[str],
    uris_to_load: set[str],
    str_suffix: str = "-logits_dict_t",
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Given a path to infered logits, load them into memory and return a dict that maps uris to logits"""

    uri_to_logit = {}
    for logit_file in logits_p.glob(f"*{str_suffix}.pt"):
        uri = logit_file.stem.split(f"{str_suffix}")[0]
        # NOTE - only load if uri in uris_to_load
        if uri in uris_to_load:
            logit_dict = torch.load(logit_file, map_location="cpu")
            # (frames, n_labels)
            uri_to_logit[uri] = torch.stack(
                [logit_dict[label] for label in labels], dim=1
            )
    return uri_to_logit


def load_gt_as_logits(
    rttm_path: Path,
    uris_to_load: set[str],
    labels: list[str],
) -> None:
    """Given a path to rttm files and a list of uris to select, loads the content of the RTTM files, converts them to tensors and returns a mapping from uris to tensors"""
    # NOTE - use `rttm_to_tensor` to load all gt rttms (filter by uris: val.txt)
    # NOTE - stack all logits
    uri_to_logit = {
        rttm_p.stem: rttm_to_tensor(rttm_p, labels=labels)
        for rttm_p in rttm_path.glob("*.rttm")
        if rttm_p.stem in uris_to_load
    }
    return uri_to_logit


def get_set(
    true_path: str | Path,
    pred_path: str | Path,
    labels: list[str],
    uri_txt: str = "val.txt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get dataset from disk.

    Args:
        true_path (str | Path): SegmaFileDataset formated path with a `.txt` file containing the uris
        pred_path (str | Path): Path to a folder containing a logits foldes
        labels (list[str]): list of labels to consider.
        uri_txt (str, optional): list of uris to load obtained from the dataset at `true_path`. Defaults to "val.txt".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: y_true, y_pred tensors.
    """
    true_path, pred_path = Path(true_path), Path(pred_path)

    uris = set(load_uris((true_path / uri_txt).with_suffix(".txt")))
    # NOTE - load load_pred_logits
    pred_logits = load_pred_logits(
        logits_p=pred_path,
        uris_to_load=uris,
        labels=labels,
    )
    # NOTE - load load_gt_as_logits
    gt_logits = load_gt_as_logits(
        rttm_path=true_path / "rttm", uris_to_load=uris, labels=labels
    )

    pred_logits_t, gt_logits_t = unify(pred_logits, gt_logits, uris_to_load=uris)
    return gt_logits_t, pred_logits_t


def get_data(
    val_true_path: Path | str,
    val_pred_path: Path | str,
    labels: list[str],
    test_true_path: Path | str | None = None,
    test_pred_path: Path | str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Retrieve the ground truth data and inferred logits files from disk.

    Args:
        val_true_path (Path | str): Path to a `SegmeFileDataset` formated dataset.
        val_pred_path (Path | str): Path to a folder containing logits as `.pt` files, one per uri.
        labels (list[str]): List of labels to consider.
        test_true_path (Path | str | None, optional): Path to a `SegmeFileDataset` formated dataset containing the test data. Only used for evaluating the thresholds after tuning. Defaults to None.
        test_pred_path (Path | str | None, optional): Path to a folder containing logits as `.pt` files, one per uri. Only used for evaluating the thresholds after tuning. Defaults to None.

    Returns:
        dict[str, dict[str, torch.Tensor]]: Returns a dict mapping dataset to ground truth and predicted logit tensor.
    """
    data_d = {}
    val_y_true_t, val_y_pred_t = get_set(
        true_path=val_true_path,
        pred_path=val_pred_path,
        labels=labels,
        uri_txt="val",
    )
    data_d["val"] = {
        "true": val_y_true_t,
        "pred": val_y_pred_t,
    }

    if test_true_path and test_pred_path:
        test_y_true_t, test_y_pred_t = get_set(
            true_path=test_true_path,
            pred_path=test_pred_path,
            labels=labels,
            uri_txt="test",
        )
        data_d["test"] = {
            "true": test_y_true_t,
            "pred": test_y_pred_t,
        }

    return data_d


def tune_multilabel(
    data_t: dict[str, dict[str, torch.Tensor]],
    thresholds: list[int] | torch.Tensor,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Tuning of the decision thresholds for a multilabel problem.
    The tuning is made using a simple grid-search given a list of thresholds to evaluate. Only the onset / lower bound is tuned

    ...
    |----LxxxxU---|
    ...

    """
    labels_to_thresh_to_score = {label: {} for label in labels}
    for thresh in tqdm(thresholds):
        f1_score = sklearn.metrics.f1_score(
            y_true=data_t["val"]["true"],
            y_pred=data_t["val"]["pred"].sigmoid() > thresh,
            average=None,
            labels=list(
                range(len(labels))
            ),  # For multilabel targets, labels are column indices.
            zero_division=1.0,
        )
        for i, label in enumerate(labels):
            labels_to_thresh_to_score[label][thresh] = f1_score[i]

    # NOTE - retrieve best thresholds
    best_thresholds = {
        label: {
            "lower_bound": round(
                float(
                    max(
                        labels_to_thresh_to_score[label],
                        key=labels_to_thresh_to_score[label].get,
                    )
                ),
                int(math.log10(n_steps)),
            ),
            "upper_bound": 1.0,
        }
        for label in labels_to_thresh_to_score.keys()
    }
    return best_thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/segma/config/default.yml"),
        help="Config file to be loaded and used for inference.",
    )
    parser.add_argument("--precision", type=float, default=0.1)
    parser.add_argument(
        "--val-ds",
        type=Path,
        default=Path("data/baby_train"),
        help="Path to the validation dataset.",
    )
    parser.add_argument(
        "--val-logits",
        type=Path,
        help="Path to the validation set logits used to tune the thresholds.",
    )
    parser.add_argument(
        "--output",
        default="tune_out",
        type=Path,
        help="Output folder of the tuned thresholds.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    assert args.precision in (0.1, 0.01)

    n_steps = int(1 / args.precision)
    thresholds = torch.linspace(0, 1, steps=n_steps).round(
        decimals=int(math.log10(n_steps))
    )

    print("[log] - Loading data...")
    data_t = get_data(
        val_true_path=args.val_ds,
        val_pred_path=args.val_logits,
        labels=config.data.classes,
    )

    print("[log] - Searching for optimal thresholds...")
    best_thresholds = tune_multilabel(data_t, thresholds, config.data.classes)

    print(f"[log] - Best threshold found")
    pprint(best_thresholds)

    args.output.mkdir(parents=True, exist_ok=True)
    YAML().dump(best_thresholds, args.output / "best_thresholds.yml")
