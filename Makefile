.PHONY: base f format tc type-check t test debug profile z zip uz unzip c clean it


base: f t

f format:
	@uv run ruff check --select I --fix
	@uv run ruff format
	@uv run ruff check

stats:
	@uv run ruff check --statistics

add-noqa:
	@echo "adds noqa to all failing lines"
	@echo "too breaking to add to the project"
	uv run ruff check --add-noqa

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
	rm -rf *.tar.gz
	rm -rf .DS_STORE

line-count:
	find src tests scripts -name '*.py' | xargs wc -l