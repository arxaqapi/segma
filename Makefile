.PHONY: base f format tc type-check t test c clean


base: f tc

f format:
	@uv run ruff check --select I --fix
	@uv run ruff format
	@uv run ruff check

tc type-check:
	@uv run mypy --disallow-untyped-defs .

t test:
	@uv run pytest -s

c clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find src -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf lightning_logs checkpoints wandb
