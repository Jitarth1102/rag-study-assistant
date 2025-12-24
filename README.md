# RAG Study Assistant (skeleton)

Local-first multimodal RAG study assistant. Runnable skeleton (CLI + Streamlit) with config, logging, DB, and tests.

## Local-first defaults

- LLM: **Ollama** (`llm.provider=ollama`, `model=llama3.1:8b`)
- Embeddings: **local** sentence-transformers (`intfloat/multilingual-e5-small`, normalized)
- OCR: **tesseract** by default (`ingest.ocr_engine=tesseract`); set `ingest.ocr_engine=auto` to use Paddle → Tesseract → Stub fallback
- Retrieval: **Qdrant** required (Docker) at `http://localhost:6333`
- OpenAI: **optional** (only if you switch providers)

## Quickstart

1) Create virtualenv (uv):
   ```bash
   make venv   # or: uv venv --python 3.11
   ```
2) Install deps (editable + dev):
   ```bash
   make install   # or: uv pip install -e ".[dev]"
   ```
3) Start Qdrant (Docker):
   ```bash
   make qdrant
   ```
4) Start Ollama (for default LLM):
   ```bash
   ollama serve
   ```
5) Launch Streamlit UI:
   ```bash
   make ui   # or: uv run streamlit run apps/streamlit/Home.py
   ```
6) CLI helpers:
   ```bash
   uv run python -m rag_assistant --help
   uv run python -m rag_assistant subjects --create "My Course"
   uv run python -m rag_assistant ingest --subject <subject_id>
   ```
7) Run tests:
   ```bash
   uv run pytest -q   # or: make test
   ```

## Configuration

- Defaults live in `config/default.yaml`.
- Override via environment variables or a `.env` file (see `.env.example`).
- Key fields: `app.data_root`, `database.sqlite_path`, `qdrant.url`/`collection`, `ingest.ocr_engine`, `llm.provider/model/base_url`, `embeddings.provider/model/vector_size`, `retrieval.top_k/min_score`.

## Usage notes

- Create subjects on Home, select one in the sidebar, then upload files on Upload. Files are stored under `data/subjects/<subject_id>/raw/` and tracked in SQLite.
- Indexing pipeline runs from Upload via **Index new uploads** (render pages, OCR, chunk, embed, upsert to Qdrant). Ensure Qdrant is running (`make qdrant`) and Ollama is up if using the default LLM.
- Chat page performs notes-only retrieval QA over indexed chunks.
- System Health page surfaces quick dependency checks (DB, Qdrant, Ollama, OCR self-test).
- CLI ingest mirrors the Upload “Index new uploads” button: `uv run python -m rag_assistant ingest --subject <subject_id>`.
- Optional web search fallback is **off by default**; enable with `WEB_ENABLED=true` and provider API key (SerpAPI). When enabled, the answer may include web citations.

## Web search (optional, off by default)

- Set env vars (or config) to enable:
  ```bash
  export WEB_ENABLED=true
  export WEB_PROVIDER=serpapi
  export WEB_API_KEY=your_key
  ```
- Behavior: only used when RAG context is weak; adds web snippets and separate web citations.
- External sources are not trusted by default; answers distinguish slide vs web citations.

## Optional OpenAI usage

- Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`).
- Switch providers via env vars, e.g.:
  ```bash
  export LLM_PROVIDER=openai
  export EMBEDDINGS_PROVIDER=openai
  export LLM_MODEL=gpt-4.1-mini
  export LLM_EMBED_MODEL=text-embedding-3-small
  ```
- Ensure `embeddings.vector_size` matches the embedding model and Qdrant collection size.

## Troubleshooting

- Qdrant container already exists: `docker rm -f qdrant_rag` then `make qdrant`.
- Qdrant port in use: set `QDRANT_PORT` before `make qdrant` or stop the conflicting service.
- Ollama port in use: stop the other process or change `llm.base_url`.
- Missing Tesseract tessdata/lang: install tesseract and language packs, or set `TESSDATA_DIR`/`ingest.tesseract_cmd`; use Upload page OCR debug/self-test to verify.

## Reset a single asset (dev/debug)
Use this to clean up generated artifacts for one asset (e.g., re-index after OCR/debugging):
```bash
./scripts/reset_asset.sh <asset_id>
```
This keeps the raw upload but removes derived pages/OCR/chunks, resets DB status, and best-effort deletes Qdrant vectors.

## Reset all assets (dev/debug)
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
