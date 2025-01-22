# find models/last -maxdepth 1 -type l -exec unlink {} \;
data_folder="data/debug_50"

source .venv/bin/activate
uv run scripts/train.py --config scripts/debug.yml --tags debug

uv run scripts/predict.py \
	--config scripts/debug.yml \
	--uris $data_folder/test.txt \
	--wavs $data_folder/wav \
	--ckpt models/last/best.ckpt

source .venv_inference/bin/activate
python scripts/evaluate.py \
	--config scripts/debug.yml \
    --gt $data_folder/rttm \
    --pred models/last/rttm
