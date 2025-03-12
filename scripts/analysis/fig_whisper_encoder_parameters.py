from pathlib import Path

import matplotlib.pyplot as plt

from segma.config import load_config
from segma.models.whisper.surgical_hydra import SurgicalHydra
from segma.utils.encoders import MultiLabelEncoder

if __name__ == "__main__":
    base = Path("import/experiments")
    for model in (
        "250210-233627-durgan",
        "250210-233627-rambou-lune",
        "250210-233627-pesce-rato",
    ):
        model = SurgicalHydra.load_from_checkpoint(
            checkpoint_path=base / model / "checkpoints/last.ckpt",
            config=load_config(base / model / "config.yml"),
            label_encoder=MultiLabelEncoder(["KCHI", "OCH", "FEM", "MAL"]),
        )
        weights = model.layer_weights.softmax(dim=0).detach().cpu().numpy()

        fig, ax = plt.subplots()

        x = list(range(1, len(weights) + 1))

        ax.bar(x, weights)
        ax.grid(axis="y")
        ax.set_xticks(x)
        ax.get_xticklabels(x)
        enc = "_".join(model.config.model.config.encoder.split("_")[:-1])
        ax.set_title(f"weight importance per whisper encoder layer\nfor {enc}")
        fig.savefig(f"weights_{enc}", dpi=300)
