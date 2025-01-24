#!/bin/bash
#SBATCH --job-name=segma_train             # Job name
#SBATCH --partition=gpu                    # Take a node from the 'gpu' partition
#SBATCH --export=ALL                       # Export your environment to the compute node
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --mem=100G                         # ram
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-train.out

# load python virtualenv
source .venv/bin/activate
module load audio-tools
module load uv

uv run scripts/train.py --config config.yml --tags whisperidou, 4s
