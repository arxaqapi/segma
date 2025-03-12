import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fig1 import set_mpl_params
from interlap import InterLap
from matplotlib.lines import Line2D

from segma.annotation import AudioAnnotation


def reverse(
    estimate: np.array, start_s: float, end_s: float, audio_length_s: float = 120.0
):
    # 7112 for 120s
    N = estimate.shape[0]

    start_idx = int(N * start_s / audio_length_s)  # = 0.5N
    end_idx = int(N * end_s / audio_length_s)  # = 0.8N
    return estimate[start_idx:end_idx]  # .mean()


def group_by_child(ier_table: dict, ier_metrics: list[str]):
    per_child: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in ier_table:
        uri = row["uri"]
        child_uri = "_".join(uri.split("_")[:2])
        for ier_m in ier_metrics:
            if row[ier_m]:
                per_child[child_uri][ier_m].append(float(row[ier_m]))
    table = []
    for child, m_to_vals in per_child.items():
        table.append(
            {"uri": child}
            | {
                metric_name: np.nanmean(values)
                for metric_name, values in m_to_vals.items()
            }
        )
    return table


def group_brouhaha_out_by_child(uri_to_brouhaha: dict[str, float]):
    # uri_to_brouhaha
    new = defaultdict(list)
    for uri, value in uri_to_brouhaha.items():
        child_uri = "_".join(uri.split("_")[:2])
        new[child_uri].append(value)

    return {uri: np.nanmean(values) for uri, values in new.items()}


@mpl.rc_context(
    set_mpl_params(
        {
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.top": True,
            # "axes.spines.bottom": False,
            # "axes.spines.left": False,
            # "axes.spines.right": False,
            # "axes.spines.top": False,
            "legend.edgecolor": (1, 1, 1, 0),
            "legend.framealpha": 0,
            "font.size": 18,
            "axes.labelsize": 20,
        }
    )
)
def fig_3(per_child: bool = False, large: bool = True, use_percentage: bool = True):
    ds: Path = Path("data/heldout")
    heldout_brouhaha: Path = ds / "brouhaha"
    # NOTE - load brouhaha
    uri_to_c50_raw: dict[str, np.ndarray] = {}
    for folder in heldout_brouhaha.glob("*"):
        c50_f = list((folder / "c50").glob("*.npy"))
        assert len(c50_f) == 1
        with c50_f[0].open("rb") as bf:
            uri_to_c50_raw[c50_f[0].stem] = np.load(bf)

    uri_to_snr_raw: dict[str, np.ndarray] = {}
    for folder in heldout_brouhaha.glob("*"):
        snr_f = list((folder / "detailed_snr_labels").glob("*.npy"))
        assert len(snr_f) == 1
        with snr_f[0].open("rb") as bf:
            uri_to_snr_raw[snr_f[0].stem] = np.load(bf)

    assert len(set(uri_to_c50_raw.keys()) & set(uri_to_snr_raw.keys())) == 600

    uri_to_rttm_gt: dict[str, list[AudioAnnotation]] = {}
    for rttm_f in (ds / "rttm").glob("*.rttm"):
        with rttm_f.open("r") as f:
            uri_to_rttm_gt[rttm_f.stem] = [
                AudioAnnotation.from_rttm(line) for line in f.readlines()
            ]

    assert set(uri_to_rttm_gt.keys()) == set(uri_to_snr_raw.keys())
    all_uris = set(uri_to_rttm_gt.keys())

    # NOTE - for each file, get all speech segments (GT) (intervals)
    # NOTE - reverse c50 / snr from segments
    uri_to_c50_mean: dict[str, float] = {}
    uri_to_snr_mean: dict[str, float] = {}

    for uri in all_uris:
        # NOTE - using rttms, get list of intervals (in s.)
        intervals = InterLap(
            [(rttm.start_time_s, rttm.end_time_s) for rttm in uri_to_rttm_gt[uri]]
        )

        # NOTE - retrieve list of values per interval
        brou_measures = {
            "c50": [],
            "snr": [],
        }
        for interval in intervals:
            start, end = interval
            brou_measures["c50"].extend(
                reverse(uri_to_c50_raw[uri], start, end).tolist()
            )
            brou_measures["snr"].extend(
                reverse(uri_to_snr_raw[uri], start, end).tolist()
            )

        # NOTE - mean of all values for snr and c50
        uri_to_c50_mean[uri] = np.mean(brou_measures["c50"])
        uri_to_snr_mean[uri] = np.mean(brou_measures["snr"])

    if use_percentage:
        ier_metrics = ("correct%", "confusion%", "false alarm%", "missed detection%")
    else:
        ier_metrics = ("correct", "confusion", "false alarm", "missed detection")
    iers_vtc = list(
        csv.DictReader(
            Path(
                "/home/tkunze/dev/vtc3/models/20240807_004906--zlzkimxc/out_heldout_analysis/ier_analysis.csv"
            ).open("r")
        )
    )
    iers_segma = list(
        csv.DictReader(
            Path(
                # "experiments/250215-174822-colin-barbu/out_heldout_analysis/ier.analysis.csv"
                "models/20250202_233503-jot6e570/out_heldout_analysis/ier_analysis.csv"
            ).open("r")
        )
    )
    # REVIEW - group by child
    if per_child:
        iers_vtc = group_by_child(iers_vtc, ier_metrics)
        iers_segma = group_by_child(iers_segma, ier_metrics)
        uri_to_snr_mean = group_brouhaha_out_by_child(uri_to_snr_mean)
        uri_to_c50_mean = group_brouhaha_out_by_child(uri_to_c50_mean)

    uris_to_use = set([line["uri"] for line in iers_vtc]) & set(
        [line["uri"] for line in iers_segma]
    )
    assert len(uris_to_use) > 40

    brouhaha_metrics = ("C50", "SNR")
    # NOTE Large
    if large:
        fig = plt.figure(layout="constrained", figsize=(18, 8), dpi=400)
        axs = fig.subplot_mosaic(
            [
                [f"{metric}.{b_metric}" for metric in ier_metrics]
                for b_metric in brouhaha_metrics
            ]
        )
    else:
        # NOTE Long layout
        fig = plt.figure(layout="constrained", figsize=(11, 13.5), dpi=400)
        axs = fig.subplot_mosaic(
            [
                [f"{metric}.{b_metric}" for b_metric in brouhaha_metrics]
                for metric in ier_metrics
            ]
        )
    # colors = plt.get_cmap("Accent", 2)
    colors = plt.get_cmap("Dark2", 2)
    for brouhaha_metric, brouhaha_vals in zip(
        brouhaha_metrics, (uri_to_c50_mean, uri_to_snr_mean)
    ):
        for metric in ier_metrics:
            for i, (model_used, ier_list) in enumerate(
                zip(("PyanNet-VTC", "Whisper-VTC"), (iers_vtc, iers_segma))
            ):
                color = colors((i + 1) % 2)
                ax_id = f"{metric}.{brouhaha_metric}"
                xs = []
                ys = []
                for ier in ier_list:
                    if ier["uri"] != "TOTAL" and ier[metric]:
                        x = brouhaha_vals[ier["uri"]]
                        if brouhaha_metric == "SNR" and x > 25:
                            continue
                        # FIXME - dead lines fail to convert
                        y = float(ier[metric])
                        if y > 1000 and use_percentage:  # and per_child:
                            continue
                        if y > 600 and use_percentage and not per_child:
                            continue
                        xs.append(x)
                        ys.append(y)

                axs[ax_id].scatter(
                    xs,
                    ys,
                    s=45 if "Whisper" in model_used else 25,
                    # marker="o",
                    marker="+" if "Whisper" in model_used else "o",
                    color=color,
                    alpha=1,
                )
                sns.regplot(x=xs, y=ys, ax=axs[ax_id], color=color, scatter=False)
            metric_name = metric if not use_percentage else metric[:-1]  #  + " (%)"
            # axs[ax_id].set_title(f"{brouhaha_metric} as a function of {metric_name}")
            # NOTE - labels
            if large:
                axs[ax_id].set_xlabel(f"Estimated {brouhaha_metric} (dB)")
                # axs[ax_id].set_ylabel(f"Percentage {metric_name}")
                axs[ax_id].set_ylabel(f"{metric_name} (%)")

            if not large and "missed detection" in metric:
                axs[ax_id].set_xlabel(f"Estimated {brouhaha_metric} (dB)")
            if not large and brouhaha_metric == "C50":
                axs[ax_id].set_ylabel(f"{metric_name} (%)")

            axs[ax_id].grid()
    fig.align_labels()
    for a in axs.keys():
        if "missed detection" not in a:
            axs[a].set_xticklabels([])
            for tick in axs[a].xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        if "SNR" in a:
            axs[a].set_yticklabels([])
            for tick in axs[a].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=colors(0), lw=4, label="Whisper-VTC"),
        Line2D([0], [0], color=colors(1), lw=4, label="PyanNet-VTC"),
    ]
    # fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    # fig.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    # if large == False:
    # axs[f"{ier_metrics[-1]}.C50"].legend(handles=legend_elements, loc="upper right")
    fig.legend(handles=legend_elements, loc="outside upper center", ncol=2)

    # fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig_name = "paper/" + fig_3.__name__
    fig_name = fig_name + ("_by_child" if per_child else "_by_file")
    if use_percentage:
        fig_name += "_percentage"
    if not large:
        fig_name += "_uprigth"
    # fig_name + ".png"
    # fig.suptitle("Identification Error Rate (IER) components \nas a function of Brouhaha's estimates.")
    fig.savefig(
        fig_name,
        dpi=400,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    fig_3(per_child=True, large=False)
    # fig_3(per_child=True, large=True)

    # fig_3(per_child=False, large=False)
    # fig_3(per_child=False, large=True)
