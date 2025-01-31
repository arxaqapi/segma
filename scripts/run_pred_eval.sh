#!/bin/bash
#SBATCH --job-name=segma_pred_eval         # Job name
#SBATCH --partition=gpu                    # Take a node from the 'gpu' partition
##SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                         # ram
##SBATCH --exclude=puck5                 # Do not run on puck5 node
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out


# 1. predict on data
source .venv/bin/activate
module load audio-tools

model_id=20250122_234139-zgl0ahak
ckpt="epoch=18-val_loss=2.132.ckpt"

# model_id=20250123_003929-opn3jjm0
# ckpt="epoch=04-val_loss=2.133.ckpt"

uv run scripts/predict.py \
    --config models/$model_id/config.yml \
    --uris data/baby_train/test.txt \
    --wavs data/baby_train/wav \
    --ckpt models/$model_id/checkpoints/$ckpt


# 2. evaluate predictions
source .venv_inference/bin/activate

python scripts/evaluate.py \
    --gt data/baby_train/rttm \
    --pred models/$model_id/rttm \
    --config models/$model_id/config.yml
