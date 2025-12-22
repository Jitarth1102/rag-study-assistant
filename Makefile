PYTHON?=python
UV?=uv
STREAMLIT?=streamlit

venv:
	$(UV) venv --python 3.11

install:
	$(UV) pip install -e ".[dev]"

test:
	$(UV) run pytest -q

ui:
	$(UV) run $(STREAMLIT) run apps/streamlit/Home.py

cli:
	$(UV) run $(PYTHON) -m rag_assistant --help

qdrant:
	bash scripts/run_qdrant.sh
