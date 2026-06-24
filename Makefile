.PHONY: install test test-e2e lint generate-data train serve

install:
	pip install -r requirements.txt

test:
	pytest --ignore=tests/e2e

test-e2e:
	pytest tests/e2e -m e2e -v --timeout=300

lint:
	ruff check .

generate-data:
	python3 cli.py generate-data

train:
	python3 cli.py train

serve:
	python3 cli.py serve --reload
