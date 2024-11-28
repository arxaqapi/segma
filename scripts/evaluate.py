# takes a folder full of true_rttms, a folder full of pred_rttms
# pass everything trough pyannote.metrics
# output overall stats
from collections import ChainMap
from pathlib import Path

from pyannote.audio.utils.metric import MacroAverageFMeasure
from pyannote.core import Annotation
from pyannote.database.util import load_rttm

# from segma.annotation import AudioAnnotation
from segma.utils.encoders import LabelEncoder, PowersetMultiLabelEncoder


def get_model_output_as_annotations(output_path: Path):
    """return a dict that maps uri to corresponding RTTM"""
    annotations: list[dict[str, Annotation]] = [
        load_rttm(f) for f in output_path.glob("*.rttm")
    ]
    annot_dict = ChainMap(*annotations)
    return annot_dict


def eval_model_output(
    rttm_true_p: Path, rttm_pred_p: Path, label_encoder: LabelEncoder
):
    assert rttm_true_p.exists()
    assert rttm_pred_p.exists()
    metric = MacroAverageFMeasure(classes=list(label_encoder.base_labels))

    uri_to_rttm_true = get_model_output_as_annotations(rttm_true_p)
    uri_to_rttm_preds = get_model_output_as_annotations(rttm_pred_p)

    supported_uris = set(uri_to_rttm_true.keys()) & set(uri_to_rttm_preds.keys())

    # TODO - base yourself on true uris and simulate empty annotations (or create empty files)
    for uri in supported_uris:
        metric(
            reference=uri_to_rttm_true[uri],
            hypothesis=uri_to_rttm_preds[uri],
            detailed=True,
        )

    try:
        metric.report(display=True).to_csv("fscore.csv")
    except BaseException as e:
        print(f"[log] - Got error runing `metric.report`: {e}")

    final_res = {}
    for label, sub_metric in metric._sub_metrics.items():
        final_res[label] = abs(sub_metric)

    print("==========================")
    print("[log] - Results")
    print(f"{abs(metric)=}")
    print("==========================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/debug/rttm")
    parser.add_argument("--pred", default="segma_out/rttm")

    args = parser.parse_args()

    args.gt = Path(args.gt) if not isinstance(args.gt, Path) else args.gt
    args.pred = Path(args.pred) if not isinstance(args.pred, Path) else args.pred

    # TODO - fixme - use config
    eval_model_output(
        rttm_true_p=args.gt,
        rttm_pred_p=args.pred,
        label_encoder=PowersetMultiLabelEncoder(
            labels=[
                "male",
                "female",
                "key_child",
                "other_child",
            ]
        ),
    )
