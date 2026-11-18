.PHONY: help setup test lint format run clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make setup    Install dependencies"
	@echo "  make test     Run full pytest test suite"
	@echo "  make lint     Run ruff lint and mypy type checks"
	@echo "  make format   Run ruff format"
	@echo "  make run      Launch Streamlit web application"
	@echo "  make clean    Clean cache and temporary files"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy models preprocessing evaluation

format:
	$(PYTHON) -m ruff format .

run:
	$(PYTHON) -m streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

