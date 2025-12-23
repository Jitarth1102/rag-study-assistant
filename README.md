# RAG Study Assistant (skeleton)

Local-first multimodal RAG study assistant. This step provides a runnable skeleton (CLI + Streamlit) with config, logging, DB stubs, and tests.

## Quickstart

1) Create virtualenv with uv:
   ```bash
   uv venv --python 3.11
   ```
2) Install deps (editable + dev):
   ```bash
   uv pip install -e ".[dev]"
   ```
3) Run CLI help:
   ```bash
   uv run python -m rag_assistant --help
   ```
4) Launch Streamlit UI:
   ```bash
   uv run streamlit run apps/streamlit/Home.py
   ```
5) Run tests:
   ```bash
   uv run pytest -q
   ```

## Usage notes

- Create subjects on the Home page, select one, then upload files on the Upload page. Files are stored under `data/subjects/<subject_id>/raw/` and tracked in SQLite.
- CLI includes a `subjects` helper to list/create subjects: `uv run python -m rag_assistant subjects --create "My Course"`.
- Indexing pipeline (PDF/images) runs from the Upload page via **Index new uploads**. It renders pages, runs PaddleOCR, chunks text, embeds, and pushes to Qdrant. Qdrant should be running locally (see `make qdrant`).
- Chat page performs notes-only retrieval QA; set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) for embeddings/LLM.

### Reset a single asset (dev/debug)
Use this to clean up generated artifacts for one asset (e.g., re-index after OCR/debugging):
```bash
./scripts/reset_asset.sh <asset_id>
```
This keeps the raw upload but removes derived pages/OCR/chunks, resets DB status, and best-effort deletes Qdrant vectors.

### Reset all assets (dev/debug)
To reset indexing outputs for all assets in the DB (keeping raw uploads):
```bash
./scripts/reset_all_assets.sh
```
You can also run via CLI: `uv run python -m rag_assistant reset-all`.

## Qdrant

A helper script starts Qdrant locally via Docker on port 6333:
```bash
make qdrant
```
This uses a named volume `qdrant_rag_data`.

## Project structure

See `apps/`, `src/rag_assistant/`, `config/`, `scripts/`, and `tests/` for the skeleton components. Config defaults live in `config/default.yaml`; override via `.env` or environment variables.
