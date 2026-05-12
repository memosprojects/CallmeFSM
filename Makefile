FUNCTIONS_DEFINITION	= data/input/functions_definition.json
INPUT				= data/input/function_calling_tests.json
OUTPUT				= data/output/function_calling_results.json

all: run

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		2>&1 | tee debug.log

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache debug.log
	rm -f $(OUTPUT)

lint:
	uv run flake8 src/ --max-line-length=100 --ignore=E501,W503

lint-strict:
	uv run flake8 src/ --max-line-length=100
	uv run mypy src/ --ignore-missing-imports --strict

.PHONY: all install run debug clean lint lint-strict
