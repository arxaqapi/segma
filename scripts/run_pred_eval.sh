#!/bin/bash
#SBATCH --job-name=segma_pred_eval         # Job name
#SBATCH --partition=gpu                    # Take a node from the 'gpu' partition
## SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                         # ram
#SBATCH --exclude=puck5                 # Do not run on puck5 node
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out

# srun python scripts/train.py --wandb --model pyannet --tags no_speech pyannet --dataset baby_train
# srun python scripts/train.py --wandb --model whisperimax --tags no_speech whisperimax --dataset baby_train



# 1. predict on data
source .venv/bin/activate
module load audio-tools

srun python scripts/predict.py \
    --uris data/baby_train/test.txt \
    --wavs data/baby_train/wav \
    --ckpt models/20241204_014516--duwjqyou/checkpoints/last.ckpt
deactivate

# 2. evaluate predictions
source .venv_inference/bin/activate

srun python scripts/evaluate.py \
    --gt data/baby_train/rttm \
    --pred models/20241204_014516--duwjqyou/rttm
