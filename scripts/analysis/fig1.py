from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from pyannote.core import Annotation
from pyannote.database.util import load_rttm

# TODO - list of models per category
models_p_d = {
    "vtc_from_scratch": (
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-dafasodi",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-kizexufi",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-qidelihu",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-vomebisa",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-vobaraji",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-kegadicu",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-cayigibo",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-denakope",
        Path("/home/tkunze/dev/vtc3/import/models") / "250216-015235-yonukoyi",
    ),
    "whisper_ssl_ds": (
        Path("import/experiments") / "250210-213001-grue-moine",
        Path("models") / "250209163425-saumon-rouge",
        Path("import/experiments") / "250210-213001-espadron",
        Path("import/experiments") / "250210-213001-arrose-noir",
        Path("import/experiments") / "250210-213001-telescopes",
        Path("import/experiments") / "250210-213001-toui-para",
        Path("import/experiments") / "250210-213001-muges",
        Path("import/experiments") / "250210-213001-corb-noir",
        Path("import/experiments") / "250210-213001-faux-fletan",
    ),
    "whisper_enc_size": (
        Path("import/experiments") / "250210-233627-durgan",
        Path("import/experiments") / "250210-233627-rambou-lune",
        Path("import/experiments") / "250210-233627-pesce-rato",
        Path("import/experiments") / "250210-233627-losera",
        Path("import/experiments") / "250210-233627-oursin-coeur",
    ),
}


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


def set_mpl_params(*args) -> dict:
    """https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams"""
    params = {
        "legend.framealpha": 1,
        "legend.facecolor": "inherit",
        "legend.edgecolor": (0.5, 0.5, 0.5),
        "legend.fancybox": False,  # round edges
        # NOTE - axes
        # "axes.spines.bottom": False,
        # "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        # NOTE - grid
        "grid.color": (0.5, 0.5, 0.5, 0.4),
        # NOTE - Text
        "font.size": 10,
        "axes.labelsize": 12,
        # "text.usetex": True,
        # "axes.prop_cycle": cycler(color=("red",))
    }
    # for param_k, param_val in params.items():
    #     mpl.rcParams[param_k] = param_val
    params.update(*args)
    return params


@mpl.rc_context(set_mpl_params({"axes.spines.left": False}))
def fig_1_whisper_size_vs_pyannet():
    """- https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_colors.html"""
    # NOTE - vtc old results
    # res_marvin_test = {"KCHI": 77.3, "OCH": 25.6, "MAL": 42.2, "FEM": 82.4}
    res_heldout_test = {"KCHI": 68.7, "OCH": 33.2, "MAL": 42.9, "FEM": 63.4}
    # avg_marvin_test = sum(res_marvin_test.values()) / 4
    avg_heldout_test = sum(res_heldout_test.values()) / 4
    # sum(res_marvin_test.values()) / 4  =  56.875
    # sum(res_heldout_test.values()) / 4 =  52.05

    whisp_to_fmeasure = {}
    # test_set = "out_marvin_test_analysis"
    test_set = "out_heldout_analysis"
    for exp in (
        "250210-233627-durgan",
        "250210-233627-rambou-lune",
        "250210-233627-pesce-rato",
        # "250210-233627-losera",
        # "250210-233627-oursin-coeur",
    ):
        model_p = Path("import/experiments") / exp
        with (model_p / "config.yml").open("r") as f:
            enc_name = yaml.safe_load(f)["model"]["config"]["encoder"].split("_")[1]

        # TODO - get fscore per label
        with (model_p / test_set / "fscore.csv").open("r") as f:
            content = f.readlines()
            # ['Macro F-measure', 'KCHI', 'OCH', 'MAL', 'FEM']
            _head = content[0].strip().split(",")[1:]
            res = content[-1].strip().split(",")[1:]

        whisp_to_fmeasure[enc_name] = float(res[0])

    fig, ax = plt.subplots(
        figsize=[6.4 * 0.7, 4.8 * 0.6]
    )  # Create a figure containing a single Axes.

    x_val = ("PyanNet", "tiny", "base", "small")  # , "medium", "large")
    # y_val = [avg_marvin_test if "marvin" in test_set else avg_heldout_test] + [
    y_val = [avg_heldout_test] + [whisp_to_fmeasure[size] for size in x_val[1:]]

    # colors = plt.get_cmap("Accent", 2)
    colors = plt.get_cmap("Dark2", 2)

    whisp_col = colors(0)
    pyann_col = colors(1)
    bar_colors = [pyann_col] + [whisp_col] * (len(x_val) - 1)

    bar_labels = ["PyanNet-VTC", "Whisper-VTC"] + ["_Whisper-VTC"] * (len(x_val) - 2)

    ax.bar(
        x_val,
        y_val,
        label=bar_labels,
        color=bar_colors,
        alpha=0.6,
        edgecolor=["black"] + ["green"] * (len(x_val) - 1),
        hatch=["/"] + [" "] * (len(x_val) - 1),
    )

    # ax.axhline(y=avg_heldout_test, ls="--", color="black")
    # ax.axhline(y=whisp_to_fmeasure["small"], ls="--", color="black")

    # NOTE - styling
    limit = 60
    ax.set_ylim(0, limit)
    y_ticks = list(
        range(0, limit + 1, 10)
    )  #  + [avg_heldout_test, whisp_to_fmeasure["small"]]
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", ls="--")
    ax.set()

    # https://stackoverflow.com/questions/35320437/drawing-brackets-over-plot
    ax.annotate(
        "Whisper-VTC",
        # xy=(0.62, -0.14),
        # xytext=(0.62, -0.18),
        xy=(0.62, 0.95),
        xytext=(0.62, 1),
        xycoords="axes fraction",
        fontsize=10,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="square", fc="white", color="w"),
        arrowprops=dict(arrowstyle="-[, widthB=6, lengthB=.5", lw=1.0, color="k"),
        # arrowprops=dict(arrowstyle="-[, widthB=8.5, lengthB=.9", lw=1.0, color="k"),
        annotation_clip=False,
    )

    # NOTE rest
    ax.set_ylabel("Average F-score (%)")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        "paper/" + fig_1_whisper_size_vs_pyannet.__name__ + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # https://github.com/jbmouret/matplotlib_for_papers
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html


if __name__ == "__main__":
    fig_1_whisper_size_vs_pyannet()
