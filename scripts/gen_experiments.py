from pathlib import Path

from segma.utils.experiment import Experiment


def meta_run_gen(
    experiments_launch_scripts: list[str | Path], meta_run_p: Path = Path("meta_run.sh")
) -> None:
    meta_run_p.unlink(missing_ok=True)
    with meta_run_p.open("w") as f:
        f.writelines(
            ["sbatch " + str(exp_p) + "\n" for exp_p in experiments_launch_scripts]
        )


if __name__ == "__main__":
    base_c = ["offline=true"]

    individual_experiments = [
        ["model.config.encoder=whiper_base_encoder"],
        ["model.config.encoder=whisper_large_encoder"],
    ]

    run_paths = []
    for exp in individual_experiments:
        extra_args = base_c + exp
        # NOTE gen experiments
        exp = Experiment(args=extra_args, tags=["test_tags", "azeaze"])
        # NOTE gen single bash script with all sbatch ... directives called `meta_run.sh`
        rp = exp.gen("jz.a100", auto_train=True)
        run_paths.append(rp)

    # NOTE - meta gen
    meta_run_gen(run_paths)
