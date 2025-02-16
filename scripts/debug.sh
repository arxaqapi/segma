# find models/last -maxdepth 1 -type l -exec unlink {} \;
data_folder="data/debug"
base_path=models
run_id=250216_xxxxxx
config="scripts/debug.yml"

# TORCH_LOGS="recompiles"
source .venv/bin/activate
uv run scripts/auto_train.py \
	--config $config \
	--tags debug \
	--run-id $run_id \
	--output $base_path

uv run scripts/predict.py \
	--config $config \
	--uris $data_folder/val.txt \
	--wavs $data_folder/wav \
	--ckpt $base_path/$run_id/checkpoints/last.ckpt \
	--output $base_path/$run_id/out_val \
# 	--save-logits

# NOTE - Tune pipeline
source .venv_inference/bin/activate
python scripts/tune.py \
	--config $config \
	--logits $base_path/$run_id/out_val/logits \
	--n-trials 10 \
	--dataset "data/debug" \
	--output $base_path/$run_id

source .venv_inference/bin/activate
python scripts/evaluate.py \
	--config $config \
    --gt $data_folder/rttm \
	--pred $base_path/$run_id/out_val/rttm
# --pred models/last/rttm
