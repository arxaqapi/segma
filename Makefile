.PHONY: base f format tc type-check t test c clean z zip uz unzip


base: f tc

f format:
	@uv run ruff check --select I --fix
	@uv run ruff format
	@uv run ruff check

tc type-check:
	@uv run mypy --disallow-untyped-defs .

t test:
	@uv run pytest -s

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