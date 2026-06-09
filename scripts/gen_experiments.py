from datetime import datetime
from pathlib import Path
from typing import Literal

import tomlkit

from segma.config import config_from_mapping
from segma.utils.experiment import new_experiment_id


def meta_run_gen(
    experiments: list[Path], meta_run_p: Path = Path("meta_run.sh")
) -> None:
    if meta_run_p.exists():
        meta_run_p.rename(
            datetime.now().strftime("%y%m%d-%H%M%S_") + "meta_run.backup.sh"
        )
    with meta_run_p.open("w") as f:
        f.writelines(["sbatch " + str(exp / "total.sh") + "\n" for exp in experiments])


def create_total_bash_script(
    output_path: Path,
    dataset_used: Path,
    job_name: str = "vtc2.1",
    device: Literal["p6", "p7"] = "p7",
):
    if device == "p6":
        partition = "cristia"
        account = "laac"
    elif device == "p7":
        partition = "dupoux"
        account = "coml"
    (output_path / "logs").mkdir(parents=True)
    (output_path / "total.sh").write_text(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=erc-{partition}
#SBATCH --account={account}
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output={output_path}/logs/%j-%x.out

module purge
module load audio-tools
module load uv

data_path={dataset_used}
experiment_path={output_path}
config_path=$experiment_path/config.toml
checkpoint_path=$experiment_path/checkpoints
checkpoint_name=best.ckpt


echo $experiment_path

# =========================================
# 1. Train
uv run scripts/train.py \\
    --config $config_path \\
    --output $experiment_path

# 2. Pred
uv run scripts/infer.py \\
    --config $config_path \\
    --uris $data_path/test.txt \\
    --wavs $data_path/wav \\
    --checkpoint $checkpoint_path/$checkpoint_name \\
    --output $experiment_path/out_test \\
    --batch_size 512 \\
    --save_logits

# 3. Eval
uv run scripts/evaluate.py \\
    --gt $data_path/rttm \\
    --pred $experiment_path/out_test/rttm \\
    --config $config_path


# =========================================
# Heldout
heldout_data_path=/store/scratch/tkunze/data/heldout
# 1. Pred
uv run scripts/infer.py \\
    --config $config_path \\
    --uris $heldout_data_path/test.txt \\
    --wavs $heldout_data_path/wav \\
    --checkpoint $checkpoint_path/$checkpoint_name \\
    --output $experiment_path/out_heldout \\
    --batch_size 512 \\
    --save_logits

# 2. Eval
uv run scripts/evaluate.py \\
    --gt $heldout_data_path/rttm \\
    --pred $experiment_path/out_heldout/rttm \\
    --config $config_path


# =========================================
# =========================================
# TODO - infer on validation set
uv run scripts/infer.py \\
    --config $config_path \\
    --uris $data_path/val.txt \\
    --wavs $data_path/wav \\
    --checkpoint $checkpoint_path/best.ckpt \\
    --output $experiment_path/out_val \\
    --batch_size 512 \\
    --save_logits

uv run scripts/evaluate.py \\
    --gt $heldout_data_path/rttm \\
    --pred $experiment_path/out_val/rttm \\
    --config $config_path

# TODO - run tuning
uv run scripts/tune.py \\
    --config $config_path \\
    --val-ds $data_path \\
    --val-logits $experiment_path/out_val/logits \\
    --output $experiment_path/out_val

best_threshold_path=$experiment_path/out_val/best_thresholds.toml


# TODO - eval w. thresholds
# TODO - on test
uv run scripts/infer.py \\
    --config $config_path \\
    --uris $data_path/test.txt \\
    --wavs $data_path/wav \\
    --checkpoint $checkpoint_path/best.ckpt \\
    --output $experiment_path/out_test_thresholds \\
    --batch_size 512 \\
    --thresholds $best_threshold_path

uv run scripts/evaluate.py \\
    --gt $data_path/rttm \\
    --pred $experiment_path/out_test_thresholds/rttm \\
    --config $config_path

# TODO - on heldout
heldout_data_path=/store/scratch/tkunze/data/heldout
uv run scripts/infer.py \\
    --config $config_path \\
    --uris $heldout_data_path/test.txt \\
    --wavs $heldout_data_path/wav \\
    --checkpoint $checkpoint_path/best.ckpt \\
    --output $experiment_path/out_heldout_thresholds \\
    --batch_size 512 \\
    --thresholds $best_threshold_path

uv run scripts/evaluate.py \\
    --gt $heldout_data_path/rttm \\
    --pred $experiment_path/out_heldout_thresholds/rttm \\
    --config $config_path
""")


if __name__ == "__main__":
    configs = [
        ("vtc_bbh1.toml", "bbh1"),
        ("vtc_hubert_base.toml", "hubert-base"),
        ("vtc_hubert_large.toml", "hubert-large"),
        ("vtc_w2v2ll4300.toml", "w2v2-ll4300"),
    ]
    nnn = 3
    base_config = Path("config") / configs[nnn][0]
    destination = (
        Path("/store/scratch/tkunze/projects/bbh-is26") / configs[nnn][1] / "_long"
    )
    destination.mkdir(parents=True, exist_ok=True)

    seeds = list(range(10))

    n_expes = len(seeds)
    experiment_ids = [new_experiment_id() for _ in range(n_expes)]

    i = 0
    for seed in seeds:
        eid = experiment_ids[i]
        (destination / eid).mkdir(exist_ok=False)

        config_dict = tomlkit.loads(base_config.read_text(encoding="utf-8"))
        config_dict["logging"]["id"] = eid
        config_dict["logging"]["tags"] += [f"model:{base_config.stem.split('_')[-1]}"]

        # NOTE - set
        config_dict["train"]["max_epochs"] = 10
        config_dict["train"]["seed"] = seed

        exp_config = config_from_mapping(config_dict)
        exp_config.save(destination / eid / "config.toml")

        create_total_bash_script(
            output_path=destination / eid,
            dataset_used=exp_config.data.dataset_path,
            job_name=f"IS26_vtc-2.1_{configs[nnn][1]}",
            device="p7",
        )
        i += 1

    meta_run_gen(
        [destination / eid for eid in experiment_ids], destination / "meta_seeds.sh"
    )
