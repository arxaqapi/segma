#!/bin/bash
#SBATCH --job-name=AT_babyH2_full         # Job name
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --partition=erc-dupoux    # Specify partition
#SBATCH --gres=gpu:1
#SBATCH --mem=70G                         # ram
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out

# load python virtualenv

module load audio-tools
module load uv

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

run_id="hubert-base-hubertbaseFF"
config_model="train_surgical_hubert_hydra.yml"
user_path="/scratch2/tcharlot"
segma_path="/home/tcharlot/coml/segma"


source $segma_path/.venv/bin/activate


if [ ! -f "$user_path/checkpoints/$run_id/run.sh" ] ; then
    mkdir $user_path/checkpoints/$run_id
    mkdir $user_path/checkpoints/$run_id/logs
    cp $segma_path/scripts/run.sh $user_path/checkpoints/$run_id/run.sh
    cp $segma_path/src/segma/config/$config_model $user_path/checkpoints/$run_id/config.yml
    echo "created experiment directory and files"
fi
    
#auto_train automatically restarts if there are checkpoints
if [ ! -f $user_path/checkpoints/$run_id/"finished" ] ; then
    sbatch --dependency=afterany:$SLURM_JOBID $user_path/checkpoints/$run_id/run.sh
else
    exit 0
fi


#srun python -u scripts/auto_train.py --config src/segma/config/default.yml --tags 2 2-lstm weighted small 128b --auto-resume --run-id 250516-140623-crapet-vert wandb.offline=false wandb.project='Segma fix' wandb.name=train-fix-small train.batch_size=128 data.dataset_multiplier=2



srun uv run $segma_path/scripts/auto_train.py --auto-resume --all-weights --run-id $run_id --output $user_path/checkpoints/ --config $user_path/checkpoints/$run_id/config.yml


#srun uv run /home/tcharlot/coml/segma/scripts/auto_train.py --all-weights --auto-resume --run-id $run_id --output /scratch2/tcharlot/checkpoints/ --config /scratch2/tcharlot/checkpoints/$run_id/config.yml

# If training is not canceled due to time out, write the finished file to stop the loop.
echo "Run finished, creating 'finished' file at '$user_path/checkpoints/$run_id/'"

touch $user_path/checkpoints/$run_id/finished
