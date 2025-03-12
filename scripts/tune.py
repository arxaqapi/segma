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
from segma.dataloader import load_uris
from segma.predict import load_all_logits, predict_all_logits
from segma.utils.encoders import (
    LabelEncoder,
    MultiLabelEncoder,
    PowersetMultiLabelEncoder,
)


def load_dataset_gt(
    uri_list_p: Path, rttm_p: Path, sub_sample: int | None = None
) -> dict[str, list[AudioAnnotation]]:
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
    annotation = Annotation(uri)

    for i, rttm in enumerate(rttms):
        segment = Segment(start=rttm.start_time_s, end=rttm.end_time_s)
        annotation[segment, i] = rttm.label
    return annotation


def eval_loaded_rttms(
    rttms_true: dict[str, list[AudioAnnotation]],
    rttms_pred: dict[str, list[AudioAnnotation]],
    label_encoder: LabelEncoder,
):
    """Evaluates the performance of a model using the `MacroAverageFMeasure`from the `pyannote.metrics`
    package.
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
) -> optuna.trial.FrozenTrial:
    """Perform histerisis-tresholding tuning for MultiLabelSegmentation problems.

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
        sub_sample=300,
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
                    name=f"{label}.lower_bound",
                    low=0.0,
                    high=1.0,  # , step=0.01
                ),
                "upper_bound": trial.suggest_float(
                    name=f"{label}.upper_bound",
                    low=0.0,
                    high=1.0,  # , step=0.01
                ),
            }
            for label in label_encoder.base_labels
        }
        predictions = predict_all_logits(
            logits=logits,
            tresholds=thresholds,
            label_encoder=label_encoder,
        )

        return eval_loaded_rttms(
            rttms_true=validation_gt_rttm,
            rttms_pred=predictions,
            label_encoder=label_encoder,
        )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    # NOTE - create initial trial
    study.enqueue_trial(
        {f"{label}.lower_bound": 0.5 for label in label_encoder.base_labels}
        | {f"{label}.upper_bound": 1.0 for label in label_encoder.base_labels}
    )
    print("[log] - Aaaaaaand let's tune <<|>>")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    return study.best_trial


def treshold_dict_to_optuna(
    treshold_p_dict: dict[str, dict[str, float]], sep: str = "."
) -> dict[str, float]:
    params: dict[str, float] = {}
    for label, tresholds in treshold_p_dict.items():
        for treshold_name, value in tresholds.items():
            params[sep.join((label, treshold_name))] = value
    return params


def optuna_out_to_treshold_d(
    optuna_param_dict: dict[str, float], sep: str = "."
) -> dict[str, dict[str, float]]:
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
        "--output", required=True, help="Output folder of the tuned tresholds."
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

    tresholds = optuna_out_to_treshold_d(best_trial.params)

    with (args.output / "tresholds.yml").open("w") as f:
        yaml.safe_dump(tresholds, f)

    print(
        f"[log] - found best trial with value: {best_trial.value} and parameters: \n{tresholds=}"
    )
