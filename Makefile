.PHONY: install dev test coverage lint typecheck format run docker clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports --no-strict-optional

format:
	ruff format src/ tests/

run:
	streamlit run app/streamlit_app.py

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov coverage.xml experiments.csv
