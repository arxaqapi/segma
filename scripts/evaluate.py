# takes a folder full of true_rttms, a folder full of pred_rttms
# pass everything trough pyannote.metrics
# output overall stats
from functools import reduce
from pathlib import Path
from typing import Mapping

from pyannote.audio.utils.metric import MacroAverageFMeasure
from pyannote.core import Annotation
from pyannote.database.util import load_rttm

from segma.config import load_config
from segma.utils.encoders import (
    LabelEncoder,
    MultiLabelEncoder,
    PowersetMultiLabelEncoder,
)


def get_model_output_as_annotations(output_path: Path) -> Mapping[str, Annotation]:
    """Load the output of a model (collectioin of `.rttm` files) as `pyannote.core.Annotation` objects
    and returns a dict that maps uri to the corresponding RTTMs.
    Args:
        output_path (Path): Path to the folder containing the model outputs.
    Returns:
        dict[str, Annotation]: mapping from uris to Annotations.
    """
    annotations = {}
    for rttm in output_path.glob("*.rttm"):
        loaded_rttms = load_rttm(rttm)
        annotations[rttm.stem] = (
            Annotation(rttm.stem) if not loaded_rttms else loaded_rttms[rttm.stem]
        )
    return annotations


def eval_model_output(
    rttm_true_p: Path,
    rttm_pred_p: Path,
    label_encoder: LabelEncoder,
    scores_output: Path = Path("fscore.csv"),
):
    """Evaluates the performance of a model using the `MacroAverageFMeasure`from the `pyannote.metrics`
    package.

    This function needs the mode to have ran on the data to evaluate with the `predict(...)` function.

    Args:
        rttm_true_p (Path): A Path to a list of ground truth `.rttm` files.
        rttm_pred_p (Path): A Path to a list of predicted `.rttm` files
        label_encoder (LabelEncoder): The label encoder used during training, to be able to retrieve the labels used.
        scores_output (Path, optional): Output Path of the csv file containg the scores. Defaults to Path("fscore.csv").
    """

    if not rttm_true_p.exists() and not rttm_true_p.is_dir():
        raise FileNotFoundError(f"Folder Path '{rttm_true_p=}' not found.")
    if not rttm_pred_p.exists() and not rttm_pred_p.is_dir():
        raise FileNotFoundError(f"Folder Path '{rttm_pred_p=}' not found.")

    metric = MacroAverageFMeasure(classes=list(label_encoder.base_labels))

    uri_to_rttm_true = get_model_output_as_annotations(rttm_true_p)
    uri_to_rttm_preds = get_model_output_as_annotations(rttm_pred_p)

    supported_uris = set(uri_to_rttm_true.keys()) & set(uri_to_rttm_preds.keys())

    # TODO - base yourself on true uris and simulate empty annotations (or create empty files)
    for uri in supported_uris:
        print(f"[log] - evaluating file: '{uri}'")

        metric(
            reference=uri_to_rttm_true[uri],
            hypothesis=uri_to_rttm_preds[uri],
            # NOTE - UEM is inferred
            detailed=True,
        )

    try:
        metric.report(display=True).to_csv(str(scores_output))
        # NOTE - make a symbolic link to it in the static folder
        static_score_p = Path("models/last/fscore.csv")
        if not scores_output.absolute() == static_score_p.absolute():
            static_score_p.parent.mkdir(parents=True, exist_ok=True)
            static_score_p.unlink(missing_ok=True)
            static_score_p.symlink_to(scores_output.absolute())
    except BaseException as e:
        print(f"[log] - Got error running `metric.report`: {e}")

    # NOTE - Manual logging of metrics
    final_res = {"Total": abs(metric)}
    for label, sub_metric in metric._sub_metrics.items():
        final_res[label] = abs(sub_metric)

    print("=====================")
    print("[log] - Results\n")
    max_len = reduce(max, [len(label) for label in final_res.keys()]) + 1
    for k, fscore in final_res.items():
        print(f"{k:<{max_len}}: {round(fscore, 5)}")
    print("=====================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/debug/rttm")
    parser.add_argument("--pred", default="segma_out/rttm")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for the training.",
    )

    args = parser.parse_args()
    args.gt = Path(args.gt)
    args.pred = Path(args.pred)
    cfg = load_config(args.config)

    eval_model_output(
        rttm_true_p=args.gt,
        rttm_pred_p=args.pred,
        label_encoder=MultiLabelEncoder(labels=cfg.data.classes)
        if "hydra" in cfg.model.name
        else PowersetMultiLabelEncoder(labels=cfg.data.classes),
        scores_output=args.pred.parent / "fscore.csv",
    )
