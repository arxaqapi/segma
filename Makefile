.PHONY: base f format tc type-check t test debug profile z zip uz unzip c clean it


base: f tc

f format:
	@uv run ruff check --select I --fix
	@uv run ruff format
	@uv run ruff check

tc type-check:
	@uv run mypy --disallow-untyped-defs .

t test:
	@uv run pytest -s

debug:
	sh scripts/debug.sh

profile:
	rm -rf profile.json profile
	@uv run scalene --profile-all --gpu --outfile profile --html --json scripts/train.py --dataset debug 

z zip:
	tar -czf segma.tar.gz -X .gitignore *

uz unzip:
	tar -xzf segma.tar.gz -C segma

c clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	find src -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf lightning_logs checkpoints wandb
	rm -rf *.tar.gz
	rm -rf .DS_STORE


# Slurm
# module load uv; source env
# uv run scripts/train.py --model pyannet --dataset debug
it:
	srun --export=ALL --mem=60G --time=5:00:00 --partition=gpu --gres=gpu:1 --job-name=VTC_IT --pty bash


# fix file .venv_inference/lib/python3.12/site-packages/pyannote/database/util.py
# by replacing all `delim_whitespace=True` with `sep=" "`
fix-pd:
	sed -i 's/delim_whitespace=True/sep=" "/g' .venv_inference/lib/python3.12/site-packages/pyannote/database/util.py
