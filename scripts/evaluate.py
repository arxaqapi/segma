from pathlib import Path

import sklearn
import tomlkit
import torch

from segma.config import Config, load_config
from segma.utils import rttm_to_tensor, unify


def load_gt_as_logits(
    rttm_path: Path,
    labels: list[str],
) -> dict[str, torch.Tensor]:
    """Given a path to rttm files and a list of uris to select, loads the content of the RTTM files, converts them to tensors and returns a mapping from uris to tensors"""
    # NOTE - use `rttm_to_tensor` to load all gt rttms (filter by uris: val.txt)
    # NOTE - stack all logits
    uri_to_logit = {
        rttm_p.stem: rttm_to_tensor(rttm_p, labels=labels)
        for rttm_p in rttm_path.glob("*.rttm")
    }
    return uri_to_logit


class Evaluator:
    def __init__(self, gt_rttm_path: Path, pred_rttm_path: Path, config: Config):
        gt_logits = load_gt_as_logits(rttm_path=gt_rttm_path, labels=config.data.labels)
        pred_logits = load_gt_as_logits(pred_rttm_path, labels=config.data.labels)

        self.supported_uris = gt_logits.keys() & pred_logits.keys()
        self.gt_logits_t, self.pred_logits_t = unify(
            gt_logits, pred_logits, uris_to_load=self.supported_uris
        )

        self.config = config
        self.output_path = pred_rttm_path.parent

    def _get_f_measure(self):
        report = sklearn.metrics.classification_report(
            y_true=self.gt_logits_t,
            y_pred=self.pred_logits_t,
            # For multilabel targets, labels are column indices.
            labels=list(range(len(self.config.data.labels))),
            target_names=self.config.data.labels,
            zero_division=1.0,
            output_dict=True,
        )

        with (self.output_path / "fscore.toml").open("w") as f:
            tomlkit.dump(report, f)

        return report

    def _ml_conf_matrix(self):
        raise NotImplementedError
        _mlm = sklearn.metrics.multilabel_confusion_matrix(
            y_true=self.gt_logits_t,
            y_pred=self.pred_logits_t,
            # For multilabel targets, labels are column indices.
            labels=list(range(len(self.config.data.labels))),
        )

        # TODO - display as square of squares w. pls.subplot_mosaic(...)

    def _get_ier(self):
        raise NotImplementedError

    def evaluate(self) -> None:
        self._get_f_measure()
        # self._get_ier()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default="data/debug/rttm")
    parser.add_argument("--pred", type=Path, default="segma_out/rttm")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for the training.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    Evaluator(args.gt, args.pred, config).evaluate()
