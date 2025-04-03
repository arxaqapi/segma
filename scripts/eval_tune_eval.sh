#!/bin/bash
#SBATCH --job-name=segma-ETEEE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-pred-eval.out

##SBATCH --export=ALL
##SBATCH --exclude=puck5


data_folder="data/baby_train"
base_path=models
run_id=250216_xxxxxx
config=$base_path/$run_id/config.yml
out_f=out_val

# NOTE - 1. Evaluate on validation split to gather logits
source .venv/bin/activate
python -u scripts/predict.py \
	--config $config \
	--uris $data_folder/val.txt \
	--wavs $data_folder/wav \
	--ckpt $base_path/$run_id/checkpoints/last.ckpt \
	--output $base_path/$run_id/$out_f \
	--save-logits

# NOTE - 2. Tune using the logits
source .venv_eval/bin/activate
python -u scripts/tune.py \
	--config $config \
	--logits $base_path/$run_id/$out_f/logits \
	--n-trials 1000 \
	--dataset $data_folder \
	--output $base_path/$run_id

# NOTE - 3. Evaluate on [all] test sets with and without threshold
python -u scripts/predict.py \
	--config $config \
	--uris $data_folder/test.txt \
	--wavs $data_folder/wav \
	--ckpt $base_path/$run_id/checkpoints/last.ckpt \
	--output $base_path/$run_id/out_test_t \
	--threshold $base_path/$run_id/threshold.yml

data_folder="data/marvin_test"
python -u scripts/predict.py \
	--config $config \
	--uris $data_folder/test.txt \
	--wavs $data_folder/wav \
	--ckpt $base_path/$run_id/checkpoints/last.ckpt \
	--output $base_path/$run_id/out_test_marvin_t \
	--threshold $base_path/$run_id/threshold.yml

data_folder="data/heldout"
python -u scripts/predict.py \
	--config $config \
	--uris $data_folder/test.txt \
	--wavs $data_folder/wav \
	--ckpt $base_path/$run_id/checkpoints/last.ckpt \
	--output $base_path/$run_id/out_test_heldout_t \
	--threshold $base_path/$run_id/threshold.yml
