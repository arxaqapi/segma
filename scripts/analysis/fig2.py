from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.polynomial import Polynomial


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
    }
    params.update(*args)
    return params


# @mpl.rc_context(set_mpl_params({"axes.spines.left": False}))
def get_mean_fscores(
    dataset_path: Path,
    labels: list[str] = ["FEM", "KCHI", "MAL", "OCH"],
    reference_ds: str = "out_heldout_analysis",
    csv_name: str = "fscore.csv",
):
    assert (dataset_path / reference_ds).exists()
    assert (dataset_path / reference_ds / csv_name).exists()

    q = (
        pl.scan_csv(dataset_path / reference_ds / csv_name, skip_rows_after_header=2)
        .drop(("SPEECH", "Macro F-measure"), strict=False)
        .rename({"": "URI"})
    )
    df = q.collect()
    avg = (
        df.with_columns(pl.mean_horizontal(labels).alias("avg-no-speech"))
        .drop(labels)
        .row(by_predicate=pl.col("URI") == "TOTAL", named=True)
    )["avg-no-speech"]
    return avg


def get_fscores_per_labels(
    dataset_path: Path,
    labels: list[str] = ["FEM", "KCHI", "MAL", "OCH"],
    reference_ds: str = "out_heldout_analysis",
    csv_name: str = "fscore.csv",
):
    assert (dataset_path / reference_ds).exists()
    assert (dataset_path / reference_ds / csv_name).exists()

    q = (
        pl.scan_csv(dataset_path / reference_ds / csv_name, skip_rows_after_header=2)
        .drop(("SPEECH", "Macro F-measure"), strict=False)
        .rename({"": "URI"})
    )
    df = q.collect()

    total_fscores = df.row(by_predicate=pl.col("URI") == "TOTAL", named=True)
    total_fscores.pop("URI")

    assert (set(labels) & set(total_fscores.keys())) == set(labels)

    return total_fscores


@mpl.rc_context(
    set_mpl_params(
        {
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 12,
            "axes.labelsize": 14,
        }
    )
)
def fig_2_whisper_size_vs_pyannet_size(
    fig_name: str,
    whisper_vtc_fscores: list[float],
    pyannet_vtc_fscores: list[float],
):
    assert len(whisper_vtc_fscores) == len(pyannet_vtc_fscores)
    n_models = len(whisper_vtc_fscores)
    # ADD 100% model
    # models/20250202_233503-jot6e570: 46.84
    # average_fscores["Whisper-VTC"].append(46.84)
    # models/20240807_004906--zlzkimxc: 0.54,0.59,0.38,0.037
    # average_fscores["PyanNet-VTC"].append(38.5)

    fig, ax = plt.subplots()  # Create a figure containing a single Axes.
    # colors = plt.get_cmap("Accent", 3)
    colors = plt.get_cmap("Dark2", 2)
    for i, (scores, model_name) in enumerate(
        zip((whisper_vtc_fscores, pyannet_vtc_fscores), ("Whisper-VTC", "PyanNet-VTC"))
    ):
        # print(scores)
        # print(len(scores))
        col = colors(i)
        x = np.array(list(range(1, n_models + 1)))
        xseq = np.linspace(1, len(x), num=100)

        p = Polynomial.fit(x, scores, deg=2)
        ax.plot(xseq, p(xseq), "--", c=col, label=model_name)

        ci = 1.96 * np.std(scores) / np.sqrt(len(x))

        ax.fill_between(
            xseq,
            p(xseq) - ci,
            p(xseq) + ci,
            alpha=0.15,
            color=col,
        )

        ax.scatter(
            x,
            scores,
            marker="+" if "Whisper" in model_name else ".",
            # s=45 if "Whisper" in model_used else 25,
            lw=1.6,
            s=100,
            color=col,
        )

    ax.set_xlabel("Percentage of the dataset used")
    ax.set_ylabel("Average F-score (%)")

    # ax.set_ylim(30, 55)
    ax.set_xlim(n_models / (n_models + 1), (n_models + 0.1))
    ax.grid(ls="--")
    ax.set_xticks(list(range(1, n_models + 1)))
    ax.set_xticklabels([f"{i}0%" for i in range(1, n_models + 1)])

    ax.legend(loc="lower right", ncols=2)

    fig.savefig(
        fig_name,
        dpi=400,
        bbox_inches="tight",
        # transparent=True,
    )


@mpl.rc_context(
    set_mpl_params(
        {
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 12,
            "axes.labelsize": 14,
        }
    )
)
def fig_2_whisper_size_vs_pyannet_size_per_label(
    fig_name: str,
    whisper_vtc_fscores: dict[str, list[float]],
    pyannet_vtc_fscores: dict[str, list[float]],
):
    assert len(whisper_vtc_fscores["KCHI"]) == len(pyannet_vtc_fscores["KCHI"])
    n_models = len(whisper_vtc_fscores["KCHI"])

    fig = plt.figure(layout="constrained", figsize=(18, 12), dpi=400)
    axs = fig.subplot_mosaic([["KCHI", "OCH"], ["MAL", "FEM"]])
    # TODO - each model maps to a dict that maps label ("KCHI", "FEM", ...) to the F-score

    colors = plt.get_cmap("Dark2", 2)
    for i, (scores_dict, model_name) in enumerate(
        zip((whisper_vtc_fscores, pyannet_vtc_fscores), ("Whisper-VTC", "PyanNet-VTC"))
    ):
        col = colors(i)
        x = np.array(list(range(1, n_models + 1)))
        xseq = np.linspace(1, len(x), num=100)

        for label, scores in scores_dict.items():
            p = Polynomial.fit(x, scores, deg=2)

            axs[label].plot(xseq, p(xseq), "--", c=col, label=model_name)

            ci = 1.96 * np.std(scores) / np.sqrt(len(x))

            axs[label].fill_between(
                xseq,
                p(xseq) - ci,
                p(xseq) + ci,
                alpha=0.15,
                color=col,
            )

            axs[label].scatter(
                x,
                scores,
                marker="+" if "Whisper" in model_name else ".",
                lw=1.6,
                s=100,
                color=col,
            )

            axs[label].set_xlabel("Percentage of the dataset used")
            axs[label].set_ylabel(f"F-score (%) for {label}")

            axs[label].set_xlim(n_models / (n_models + 1), (n_models + 0.1))
            axs[label].grid(ls="--")
            axs[label].set_xticks(list(range(1, n_models + 1)))
            axs[label].set_xticklabels([f"{i}0%" for i in range(1, n_models + 1)])

            axs[label].legend(loc="lower right", ncols=2)

    fig.savefig(
        fig_name,
        dpi=400,
        bbox_inches="tight",
        # transparent=True,
    )


"""
Fixeridou plan
fig 1:
- s'assurer que l'eval des Whisper soit faite sur le marvin test set
  OU s'assurer que VTC soit bien 'heldout'

fig 3:
- s'assurer que le heldout est bien utilisé
- affichage en %
- model Whisper-VTC = 100% jusqu'a convergence (jot6e570)
- model PyanNet-VTC = non-tuned heldout

"""

if __name__ == "__main__":
    vtc_fs_base = Path("/home/tkunze/dev/vtc3/import/models")
    pyann_from_scratch = (
        "250216-015235-dafasodi",
        "250216-015235-kizexufi",
        "250216-015235-qidelihu",
        "250216-015235-vomebisa",
        "250216-015235-vobaraji",
        "250216-015235-kegadicu",
        "250216-015235-cayigibo",
        "250216-015235-denakope",
        "250216-015235-yonukoyi",
    )

    whisper_from_scratch = (
        Path("import/experiments") / "250210-213001-grue-moine",
        Path("models") / "250209163425-saumon-rouge",
        Path("import/experiments") / "250210-213001-espadron",
        Path("import/experiments") / "250210-213001-arrose-noir",
        Path("import/experiments") / "250210-213001-telescopes",
        Path("import/experiments") / "250210-213001-toui-para",
        Path("import/experiments") / "250210-213001-muges",
        Path("import/experiments") / "250210-213001-corb-noir",
        Path("import/experiments") / "250210-213001-faux-fletan",
    )
    full_whisper_vtc = Path("models/20250202_233503-jot6e570")
    full_pyannet_vtc = Path("/home/tkunze/dev/vtc3/models/20240807_004906--zlzkimxc")

    whisp_scores = [
        get_mean_fscores(model, csv_name="fscore.csv") for model in whisper_from_scratch
    ]
    pyann_scores = [
        get_mean_fscores(vtc_fs_base / model, csv_name="fscore_heldout.csv")
        for model in pyann_from_scratch
    ]

    # NOTE - output figure 2
    # fig_2_whisper_size_vs_pyannet_size(
    #     "paper/" + fig_2_whisper_size_vs_pyannet_size.__name__ + ".png",
    #     whisp_scores,
    #     pyann_scores,
    # )
    # fig_2_whisper_size_vs_pyannet_size(
    #     "paper/" + fig_2_whisper_size_vs_pyannet_size.__name__ + "_100.png",
    #     whisp_scores + [get_mean_fscores(full_whisper_vtc)],
    #     pyann_scores
    #     + [
    #         get_mean_fscores(
    #             full_pyannet_vtc,
    #             reference_ds="out_heldout_analysis_tune",
    #             csv_name="fscore_heldout.csv",
    #         )
    #     ],
    # )

    # NOTE - generate graph per class
    whisp_scores_p_label = [
        get_fscores_per_labels(model, csv_name="fscore.csv")
        for model in whisper_from_scratch
    ]
    pyann_scores_p_label = [
        get_fscores_per_labels(vtc_fs_base / model, csv_name="fscore_heldout.csv")
        for model in pyann_from_scratch
    ]
    whisp_label_to_scores = {
        label: [d[label] for d in whisp_scores_p_label]
        for label in whisp_scores_p_label[0]
    }
    pyann_label_to_scores = {
        label: [d[label] for d in pyann_scores_p_label]
        for label in pyann_scores_p_label[0]
    }

    fig_2_whisper_size_vs_pyannet_size_per_label(
        "paper/" + fig_2_whisper_size_vs_pyannet_size_per_label.__name__ + ".png",
        whisp_label_to_scores,
        pyann_label_to_scores,
    )

    print("[log] - done")
