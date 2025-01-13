find models/last -maxdepth 1 -type l -exec unlink {} \;

source .venv/bin/activate
uv run scripts/train.py --config scripts/debug.yml --tags debug

uv run scripts/predict.py \
	--config scripts/debug.yml \
	--uris data/debug/test.txt \
	--wavs data/debug/wav \
	--ckpt models/last/best.ckpt

source .venv_inference/bin/activate
python scripts/evaluate.py \
	--config scripts/debug.yml \
    --gt data/debug/rttm \
    --pred models/last/rttm
