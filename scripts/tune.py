import argparse
import random
from collections import defaultdict
from pathlib import Path

import optuna
import yaml
from pyannote.audio.utils.metric import MacroAverageFMeasure
from pyannote.core import Annotation, Segment

from segma.annotation import AudioAnnotation
from segma.config.base import Config, load_config
from segma.data.loaders import load_uris
from segma.predict import load_all_logits, predict_all_logits
from segma.utils.encoders import (
    LabelEncoder,
    MultiLabelEncoder,
    PowersetMultiLabelEncoder,
)


def load_dataset_gt(
    uri_list_p: Path, rttm_p: Path, sub_sample: int | None = None
) -> dict[str, list[AudioAnnotation]]:
    """Load ground truth annotations for a list of audio URIs.

    Args:
        uri_list_p (Path): Path to the file containing a list of URIs.
        rttm_p (Path): Directory path containing RTTM files (one file per URI).
        sub_sample (int | None, optional): If provided, randomly sample this number of URIs from the list. Defaults to None.

    Returns:
        dict[str, list[AudioAnnotation]]: Dictionary mapping each URI to its list of AudioAnnotation instances.
    """
    assert uri_list_p.exists()
    assert rttm_p.exists()

    uri_list = set(load_uris(uri_list_p))
    if sub_sample:
        uri_list = random.sample(list(uri_list), k=sub_sample)

    uri_to_rttm = {}
    for uri in uri_list:
        uri_rttm_p = (rttm_p / uri).with_suffix(".rttm")
        with uri_rttm_p.open("r") as f:
            annots = [AudioAnnotation.from_rttm(line) for line in f.readlines()]
        uri_to_rttm[uri] = annots
    return uri_to_rttm


def aa_to_annotation(uri: str, rttms: list[AudioAnnotation]) -> Annotation:
    """Convert a list of `AudioAnnotations` into a `pyannote.core.Annotation`.

    Args:
        uri (str): URI of the audio file.
        rttms (list[AudioAnnotation]): List of AudioAnnotation objects for this file.


    Returns:
        Annotation: pyannote.core.Annotation object with labeled segments.
    """
    annotation = Annotation(uri)

    for i, rttm in enumerate(rttms):
        segment = Segment(start=rttm.start_time_s, end=rttm.end_time_s)
        annotation[segment, i] = rttm.label
    return annotation


def eval_loaded_rttms(
    rttms_true: dict[str, list[AudioAnnotation]],
    rttms_pred: dict[str, list[AudioAnnotation]],
    label_encoder: LabelEncoder,
) -> float:
    """Evaluate predictions against ground truth using macro-averaged F-measure.

    This function leverage the `MacroAverageFMeasure` function from the `pyannote.metrics` package.

    Args:
        rttms_true (dict[str, list[AudioAnnotation]]): Ground truth annotations.
        rttms_pred (dict[str, list[AudioAnnotation]]): Predicted annotations.
        label_encoder (LabelEncoder): Label encoder containing the full label set.

    Returns:
        float: Metric object value after being updated with all files' scores.
    """
    metric = MacroAverageFMeasure(classes=list(label_encoder.base_labels))

    supported_uris = set(rttms_true.keys()) & set(rttms_pred.keys())
    assert len(supported_uris) > 0

    for uri in supported_uris:
        # print(f"[log] - evaluating file: '{uri}'")
        metric(
            reference=aa_to_annotation(uri, rttms_true[uri]),
            hypothesis=aa_to_annotation(uri, rttms_pred[uri]),
            # NOTE - UEM is inferred
            detailed=True,
        )
    print(f"[log] - total f1-score {abs(metric)}")
    return abs(metric)


def tune(
    logits_p: Path,
    config: Config,
    n_trials: int = 100,
    dataset_to_tune_on: Path = Path("data/baby_train"),
    sep: str = ".",
) -> optuna.trial.FrozenTrial:
    """Perform histerisis-thresholding tuning for MultiLabelSegmentation problems.

    Args:
        logits_p (Path): Path to saved logits to use for in-memory optimisation.
        config (Config): Config file of the corresponding run.
        n_trials (int, optional): Number of trials to run. Defaults to 100.
        dataset_to_tune_on (Path, optional): Path to the dataset to use, should contain a `val.txt` file. Defaults to Path("data/baby_train").

    Returns:
        optuna.trial.FrozenTrial: Best found trial.
    """

    assert (dataset_to_tune_on / "val.txt").exists()
    # NOTE - load GT validation RTTMs, ensure only the validation set is used here.
    if not (dataset_to_tune_on / "val.txt").exists():
        raise ValueError("File `val.txt` is not available in the considered dataset.")
    validation_gt_rttm = load_dataset_gt(
        uri_list_p=dataset_to_tune_on / "val.txt",
        rttm_p=dataset_to_tune_on / "rttm",
        # TODO - add percentage of the dataset as parameter
    )
    with (logits_p.parent.parent / "tune_uris.txt").open("w") as f:
        f.writelines([uri + "\n" for uri in validation_gt_rttm.keys()])

    # NOTE - load all logits test set in memory
    logits = load_all_logits(logits_p=logits_p)
    label_encoder: LabelEncoder = (
        MultiLabelEncoder(config.data.classes)
        if "hydra" in config.model.name
        else PowersetMultiLabelEncoder(config.data.classes)
    )

    def objective(trial: optuna.Trial):
        thresholds = {
            label: {
                "lower_bound": trial.suggest_float(
                    name=f"{label}{sep}lower_bound",
                    low=0.0,
                    high=1.0,  # , step=0.01
                ),
                "upper_bound": 1.0,
            }
            for label in label_encoder.base_labels
        }
        predictions = predict_all_logits(
            logits=logits,
            thresholds=thresholds,
            label_encoder=label_encoder,
        )

        return eval_loaded_rttms(
            rttms_true=validation_gt_rttm,
            rttms_pred=predictions,
            label_encoder=label_encoder,
        )

    # TODO - add sampler as param
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # NOTE - create initial trial
    study.enqueue_trial(
        {f"{label}{sep}lower_bound": 0.5 for label in label_encoder.base_labels}
        # | {f"{label}{sep}upper_bound": 1.0 for label in label_encoder.base_labels}
    )
    print("[log] - Aaaaaaand let's tune <<|>>")
    # TODO - add n_jobs as parameter
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    return study.best_trial


def threshold_dict_to_optuna(
    threshold_p_dict: dict[str, dict[str, float]], sep: str = "."
) -> dict[str, float]:
    """Transforms a nested dict with structure: `{label: {lower_bound: x, upper_bound: y}}`
    to a flat representation with `sep` as level separators: `{"label[sep]lower_bound": ..., "label[sep]upper_bound": ...}`.

    Args:
        threshold_p_dict (dict[str, dict[str, float]]): Treshold dict loaded from disc
        sep (str, optional): Separator to use for a flat representation. Defaults to ".".

    Returns:
        dict[str, float]: _description_
    """
    params: dict[str, float] = {}
    for label, thresholds in threshold_p_dict.items():
        for threshold_name, value in thresholds.items():
            params[sep.join((label, threshold_name))] = value
    return params


def optuna_out_to_threshold_d(
    optuna_param_dict: dict[str, float], sep: str = "."
) -> dict[str, dict[str, float]]:
    """Takes a flat dict with `sep` as hierarchy divider and creates a nested dict with
    structure: `{label: {lower_bound: x, upper_bound: y}}`.

    Args:
        optuna_param_dict (dict[str, float]): Flat dictionary containing the optuna thresholds.
        sep (str, optional): Separator used to create . Defaults to ".".

    Returns:
        dict[str, dict[str, float]]: mapping from labels to corresponding threshold dict.
    """
    params = defaultdict(dict)
    for value_name, value in optuna_param_dict.items():
        label, tresh_name = value_name.split(sep)
        params[label][tresh_name] = value
    return dict(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for inference.",
    )
    parser.add_argument(
        "--logits",
        type=str,
        required=True,
        help="Path to logits.",
    )
    parser.add_argument(
        "--n-trials",
        "--n_trials",
        type=int,
        default=100,
        help="Number of optuna trials to run.",
    )
    parser.add_argument(
        "--dataset",
        default="data/baby_train",
        help="Dataset to use for tuning, should contain a 'val.txt' files.",
    )
    parser.add_argument(
        "--output", required=True, help="Output folder of the tuned thresholds."
    )

    args = parser.parse_args()
    args.dataset = Path(args.dataset)
    args.output = Path(args.output)
    args.logits = Path(args.logits)
    args.output.mkdir(parents=True, exist_ok=True)

    print("[log] - starting tuning pipeline ...:)")
    best_trial = tune(
        logits_p=args.logits,
        config=load_config(args.config),
        n_trials=args.n_trials,
        dataset_to_tune_on=args.dataset,
    )

    thresholds = optuna_out_to_threshold_d(best_trial.params)

    # NOTE - fix threshnolds
    thresholds = {
        label: vals | {"upper_bound": 1.0} for label, vals in thresholds.items()
    }

    with (args.output / "thresholds.yml").open("w") as f:
        yaml.safe_dump(thresholds, f)

    print(
        f"[log] - found best trial with value: {best_trial.value} and parameters: \n{thresholds=}"
    )
