from pathlib import Path

from segma.utils.experiment import Experiment


def meta_run_gen(
    experiments: list[Experiment], meta_run_p: Path = Path("meta_run.sh")
) -> None:
    meta_run_p.unlink(missing_ok=True)
    with meta_run_p.open("w") as f:
        f.writelines(["sbatch " + str(exp.run_script_p) + "\n" for exp in experiments])

        f.writelines(["# rm -rf " + str(exp.exp_path) + "\n" for exp in experiments])


if __name__ == "__main__":
    base_c = [
        "wandb.offline=false",
        "wandb.project='Segma scale'",
        "wandb.name=train-SH-sub-data",
    ]
    base_t = ["2-lstm", "weighted", "base", "jz"]

    # model.config.encoder=whiper_base_encoder
    values_to_consider = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    individual_experiments = [
        [f"data.dataset_multiplier={val}"] for val in values_to_consider
    ]
    all_tags = [[str(val)] + base_t for val in values_to_consider]

    experiments = []
    for exp, tags in zip(individual_experiments, all_tags):
        extra_args = base_c + exp

        # NOTE gen experiments
        exp = Experiment(args=extra_args, tags=tags)

        # NOTE gen single bash script with all sbatch ... directives called `meta_run.sh`
        exp.gen("jz.a100", auto_train=True)
        experiments.append(exp)

    # NOTE - meta gen
    meta_run_gen(experiments)
