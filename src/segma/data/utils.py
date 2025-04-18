from functools import reduce
from pathlib import Path

from interlap import InterLap

from segma.annotation import AudioAnnotation
from segma.utils.conversions import seconds_to_frames


def load_uris(file_p: Path) -> list[str]:
    """Loads a list of URIs from a given text file.

    Args:
        file_p (Path): Path to the file containing one URI per line.

    Returns:
        list[str]: A list of URIs as strings.

    Example:
        Contents of the file pointed to by `file_p`:
            ```
            # file_p content
            uri_001
            uri_002
            uri_003
            ```
    """
    with file_p.open("r") as f:
        return [line.strip() for line in f.readlines()]


def load_annotations(aa_file_p: Path) -> list[AudioAnnotation]:
    """Loads audio annotations from a file.

    Args:
        aa_file_p (Path): Path to the file containing audio annotations.

    Returns:
        list[AudioAnnotation]: A list of parsed `AudioAnnotation` objects.
    """
    with aa_file_p.open("r") as f:
        return [AudioAnnotation.read_line(line) for line in f.readlines()]


def filter_annotations(
    annotations: list[AudioAnnotation],
    covered_labels: tuple[str, ...] | list[str] | set[str],
) -> list[AudioAnnotation]:
    """Filters a list of audio annotation by removing labels that are not in `covered_labels`.

    Args:
        annotations (list[AudioAnnotation]): A list of `AudioAnnotation` objects.
        covered_labels (tuple[str, ...]): tuple of labels to keep.

    Returns:
        list[AudioAnnotation]: A filtered list of `AudioAnnotation` objects.
    """
    return [annot for annot in annotations if annot.label in covered_labels]


def total_annotation_duration_ms(annotations: list[AudioAnnotation]) -> float:
    """Computed the total annotated duration in ms of a list of `AudioAnnotation` objects.

    Args:
        annotations (list[AudioAnnotation]): List of `AudioAnnotation` objects.

    Returns:
        float: Total duration in ms of all annotated segments.
    """
    return reduce(lambda b, e: b + e.duration_ms, annotations, 0)


def total_annotation_duration_f(
    annotations: list[AudioAnnotation], sample_rate: int
) -> int:
    """Computed the total annotated duration in number of frames of a list of `AudioAnnotation` objects.

    Args:
        annotations (list[AudioAnnotation]): List of `AudioAnnotation` objects.
        sample_rate (int): Sample rate of the audio, used to convert to the right amount of frames

    Returns:
        float: Total duration in ms of all annotated segments.
    """
    return seconds_to_frames(
        reduce(lambda b, e: b + e.duration_s, annotations, 0.0), sample_rate=sample_rate
    )


def create_interlap_from_annotation(annotations: list[AudioAnnotation]) -> InterLap:
    """Given a list of `AudioAnnotation`, create an `Interlap` object using the frame information

    Args:
        annotations (list[AudioAnnotation]): _description_

    Returns:
        _type_: _description_
    """
    # REVIEW - use the label_encoder(annot.label)
    return InterLap(
        [(annot.start_time_f, annot.end_time_f, annot.label) for annot in annotations]
    )
