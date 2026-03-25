import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set the seeds accross the python library random genarator, numpy and pytorch for reproductibility.

    Args:
        seed (int): seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True


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


def pad_tensors(tensor_0: torch.Tensor, tensor_1: torch.Tensor, dim: int = 0):
    """Given two tensors, pad the smallest one with zeroes to match the longest one on dimension `dim`."""
    assert len(tensor_0.shape) == 2 and len(tensor_1.shape) == 2
    assert tensor_0.shape[(dim + 1) % 2] == tensor_1.shape[(dim + 1) % 2]
    return (
        torch.nn.functional.pad(
            tensor_0, (0, 0, 0, max(0, tensor_1.shape[dim] - tensor_0.shape[dim]))
        ),
        torch.nn.functional.pad(
            tensor_1, (0, 0, 0, max(0, tensor_0.shape[dim] - tensor_1.shape[dim]))
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
