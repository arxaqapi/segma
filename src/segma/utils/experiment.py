import random
from datetime import datetime
from pathlib import Path
from typing import Literal

WORD_LIST = ("scripts/extra/names.txt", 3198)


def _get_random_word(word_list_p: Path | str, n_words: int) -> str:
    with Path(word_list_p).open("r") as f:
        return f.readlines()[random.randint(0, n_words)].strip()


def new_experiment_id():
    e_id = datetime.now().strftime("%y%m%d-%H%M%S-")
    return e_id + _get_random_word(*WORD_LIST)


class Experiment:
    def __init__(
        self,
        base: Path = Path("experiments"),
        args: list[str] = [],
        tags: list[str] = [],
    ) -> None:
        self.base = base
        self.run_id = new_experiment_id()

        i = 0
        while (self.base / self.run_id).exists():
            self.run_id = new_experiment_id()
            i += 1
            if i > 10:
                raise RuntimeError(
                    f"Experiment id: {self.run_id} at full path: {self.exp_path} already exists, aborting."
                )

        self.args = args
        self.tags = tags

    def gen(
        self,
        target: Literal["oberon", "jz.v100", "jz.a100"],
        auto_train: bool = False,
    ) -> str | Path:
        """Generate all files necessary for the experiment.
        Return the path to the generated `run.sh` script.
        """
        # NOTE - 1. Folder gen
        if self.exp_path.exists():
            raise RuntimeError(
                f"Experiment id: {self.run_id} at full path: {self.exp_path} already exists, aborting."
            )
        self.exp_path.mkdir(exist_ok=False, parents=True)

        # NOTE - tag file gen
        if self.tags:
            with (self.exp_path / "tags").open("w") as f:
                f.writelines([tag + "\n" for tag in self.tags])

        # NOTE - 2. run template resolving
        match target:
            case "oberon":
                script_content = self._slurm_oberon(auto_train=auto_train)
            case "jz.v100":
                script_content = self._slurm_jz("v100", auto_train=auto_train)
            case "jz.a100":
                script_content = self._slurm_jz("a100", auto_train=auto_train)
            case _:
                raise ValueError(f"Target `{target}` is not supported right now.")

        # NOTE - 3. write run.sh script
        with self.run_script_p.open("w") as f:
            f.write(script_content)

        if not self.run_script_p.exists():
            raise RuntimeError(
                f"Something wen wrong generating the {self.run_script_p.parts[-1]} script at path: `{self.run_script_p.parent}`."
            )

        return self.run_script_p

    @property
    def exp_path(self):
        """Full Path to the experiment folder"""
        return self.base / self.run_id

    @property
    def run_script_p(self) -> Path:
        return self.exp_path / "run.sh"

    def _slurm_oberon(self, auto_train: bool):
        """Generate an Oberon compatible run script with auto-requeue."""
        return f"""#!/bin/bash
#!/bin/bash
#SBATCH --job-name=segma_train             # Job name
#SBATCH --partition=gpu                    # Take a node from the 'gpu' partition
#SBATCH --export=ALL                       # Export your environment to the compute node
##SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --mem=100G                         # ram
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output={self.exp_path}/%j-log.txt

# load python virtualenv
source .venv/bin/activate
module load audio-tools
module load uv

{self._train_instructions(auto_train)}
"""

    def _slurm_jz(self, target: Literal["v100", "a100"], auto_train: bool):
        """Generate a Jean-Zay compatible run script with auto-requeue.

        - http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm.html
        - inspired by: https://docs.hpc.gwdg.de/how_to_use/slurm/resubmitting_jobs/index.html
        """
        match target:
            case "v100":
                gpu_target = "v100-16g"
                qos = "qos_gpu-t3"  # 20h max
                # qos = "qos_gpu-t4"  # 100h max
                extra_modules = []
            case "a100":
                gpu_target = "a100"
                qos = "qos_gpu_a100-t3"  # 20h max
                extra_modules = ["module load arch/a100"]
            case _:
                raise NotImplementedError
        # REVIEW - use default config or generate a new one ? (a backup is made by the train script)
        #        --config {self.exp_path}/config.yml \
        return f"""#!/bin/bash
#SBATCH --account=hmx@{target}
#SBATCH --job-name=segma_train

#SBATCH -C {gpu_target}
#SBATCH --qos={qos}

#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=8            # nombre de CPU par tache pour gpu_p5 (1/4 des CPU du noeud 8-GPU A100)
#SBATCH --hint=nomultithread

#SBATCH --time=20:00:00
#SBATCH --output={self.exp_path}/%j-%x-log.txt

module purge
{"\n".join(extra_modules)}

module load python/3.11.5
module load ffmpeg/6.1.1
module load sox/14.4.2

source .venv/bin/activate

{self._auto_train_instructions()[0] if auto_train else ""}

{self._train_instructions(auto_train)}

{self._auto_train_instructions()[1] if auto_train else ""}
"""

    def _auto_train_instructions(self):
        return (
            "#auto_train automatically restarts if there are checkpoints\n"
            + f'if [ ! -f {self.exp_path}/"finished" ] ; then\n'
            + f"	sbatch --dependency=afterany:$SLURM_JOBID {self.run_script_p}\n"
            + "else\n"
            + "	exit(0)\n"
            + "fi\n",
            "# If training is not canceled due to time out, write the finished file to stop the loop.\n"
            + f"""echo "Run finished, creating 'finished' file at '{self.exp_path}/'\n"""
            + f"touch {self.exp_path}/finished",
        )

    def _train_instructions(self, auto_train: bool):
        return f"""srun python -u scripts/{"auto_train" if auto_train else "train"}.py \\
    --config src/segma/config/default.yml \\
    --tags {" ".join(self.tags)} \\
    --auto-resume \\
	--run-id {self.run_id} \\
    {" ".join(self.args)}"""
