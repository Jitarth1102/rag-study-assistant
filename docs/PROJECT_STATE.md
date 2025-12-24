# Project State Report — `rag-study-assistant`

## 1) Executive Summary

This repo is a local-first “RAG Study Assistant” skeleton: users create a **subject**, upload study materials, run an **OCR→chunk→embed→Qdrant** indexing pipeline, then ask questions in a **notes-only** chat UI that retrieves relevant chunks and generates an answer using an LLM (`README.md:1`, `apps/streamlit/Home.py:24`, `apps/streamlit/pages/2_Upload.py:48`, `src/rag_assistant/ingest/pipeline.py:88`, `src/rag_assistant/rag/answerer.py:56`).

**Current user workflow (upload → index → chat)**
- **Create/select subject** (Home + sidebar) → persisted in Streamlit session state (`apps/streamlit/Home.py:29`, `src/rag_assistant/ui/sidebar.py:29`, `src/rag_assistant/ui/session_state.py:30`)
- **Upload files** → stored under `data/subjects/<subject_id>/raw/` and tracked in SQLite `assets` (`apps/streamlit/pages/2_Upload.py:39`, `src/rag_assistant/services/asset_service.py:55`, `src/rag_assistant/services/subject_service.py:57`)
- **Index new uploads** → renders pages, OCRs, chunks, embeds, and upserts to Qdrant (`apps/streamlit/pages/2_Upload.py:48`, `src/rag_assistant/ingest/pipeline.py:210`)
- **Chat** → embeds query, searches Qdrant (subject-filtered with fallback), expands context via neighbor pages, renders prompt, calls LLM, returns citations + debug (`apps/streamlit/pages/1_Chat.py:7`, `src/rag_assistant/services/chat_service.py:9`, `src/rag_assistant/rag/answerer.py:56`)

**What’s local vs external services**
- **Local (in-repo runtime)**
  - Streamlit UI (`apps/streamlit/Home.py:1`)
  - SQLite DB + migrations (`src/rag_assistant/db/sqlite.py:15`, `src/rag_assistant/db/schema.sql:1`)
  - Filesystem artifacts under `data/` (`config/default.yaml:4`, `src/rag_assistant/services/subject_service.py:57`, `src/rag_assistant/ingest/pipeline.py:98`)
  - PDF rendering via PyMuPDF (`src/rag_assistant/ingest/render/pdf_to_images.py:11`)
  - OCR via **Tesseract/PaddleOCR** (but requires external binaries/models; see below)
  - Embeddings via sentence-transformers (local model load) (`src/rag_assistant/retrieval/embedder.py:37`)
- **External processes/services**
  - **Qdrant** (expected at `qdrant.url`) (`config/default.yaml:14`, `src/rag_assistant/retrieval/vector_store/qdrant.py:101`, `scripts/run_qdrant.sh:1`)
  - **Ollama server** for LLM when `llm.provider=ollama` (`config/default.yaml:38`, `src/rag_assistant/llm/ollama_client.py:18`)
  - Optional **OpenAI API** when `llm.provider=openai` or `embeddings.provider=openai` (`src/rag_assistant/llm/provider.py:19`, `src/rag_assistant/retrieval/embedder.py:47`)

**Models/providers used (LLM + embeddings + OCR)**
- **LLM**
  - Default: `ollama` + model `llama3.1:8b` (`config/default.yaml:38`, `src/rag_assistant/llm/provider.py:19`, `src/rag_assistant/llm/ollama_client.py:18`)
  - Optional: `openai` chat via `llm.chat_model` (`config/default.yaml:43`, `src/rag_assistant/llm/provider.py:27`)
- **Embeddings**
  - Default: `local` sentence-transformers model `intfloat/multilingual-e5-small` with `normalize_embeddings=True` (`config/default.yaml:46`, `src/rag_assistant/retrieval/embedder.py:53`)
  - Optional: OpenAI embeddings via `llm.embed_model` (`config/default.yaml:44`, `src/rag_assistant/retrieval/embedder.py:61`)
- **OCR**
  - Default config selects `tesseract` (`config/default.yaml:22`, `src/rag_assistant/ingest/ocr/factory.py:15`)
  - Supported: `paddle`, `tesseract`, and `stub` fallback (`src/rag_assistant/ingest/ocr/factory.py:15`)

---

## 2) What Is Implemented So Far (Feature Inventory)

**Subjects (create/select)**
- **Where**
  - UI create/select: `apps/streamlit/Home.py:29`, `src/rag_assistant/ui/sidebar.py:29`
  - Service layer: `src/rag_assistant/services/subject_service.py:46`, `src/rag_assistant/services/subject_service.py:64`
  - CLI: `src/rag_assistant/cli.py:46`
- **Key functions/classes**
  - `subject_service.create_subject()` (`src/rag_assistant/services/subject_service.py:64`)
  - `subject_service.list_subjects()` (`src/rag_assistant/services/subject_service.py:46`)
  - `session_state.set_selected_subject()` (`src/rag_assistant/ui/session_state.py:34`)
- **Data persisted**
  - SQLite `subjects(subject_id,name,created_at,meta_json)` (`src/rag_assistant/db/schema.sql:23`)
  - Filesystem: ensures `data/subjects/<subject_id>/raw/` exists (`src/rag_assistant/services/subject_service.py:57`)

**Assets (upload/storage)**
- **Where**
  - Upload UI: `apps/streamlit/pages/2_Upload.py:39`
  - Asset service: `src/rag_assistant/services/asset_service.py:55`
- **Key functions/classes**
  - `asset_service.add_asset()` writes to disk and inserts `assets` row; dedup by content sha (`src/rag_assistant/services/asset_service.py:55`)
  - `asset_service.list_assets()` (`src/rag_assistant/services/asset_service.py:112`)
- **Data persisted**
  - SQLite `assets(...)` (`src/rag_assistant/db/schema.sql:30`)
  - Raw files under `data/subjects/<subject_id>/raw/` (`src/rag_assistant/services/subject_service.py:57`)

**Indexing pipeline (PDF render → OCR → chunking → embeddings → Qdrant upsert)**
- **Where**
  - Orchestrator: `src/rag_assistant/ingest/pipeline.py:88`
  - Render: `src/rag_assistant/ingest/render/pdf_to_images.py:11`, `src/rag_assistant/ingest/render/image_to_page.py:12`
  - OCR: `src/rag_assistant/ingest/ocr/factory.py:15`, `src/rag_assistant/ingest/ocr/paddle.py:22`, `src/rag_assistant/ingest/ocr/tesseract.py:37`
  - Chunking: `src/rag_assistant/ingest/chunking/layout_chunker.py:23`
  - Embeddings: `src/rag_assistant/retrieval/embedder.py:37`
  - Vector store: `src/rag_assistant/retrieval/vector_store/qdrant.py:101`
- **Key functions/classes**
  - `process_subject_new_assets()` (`src/rag_assistant/ingest/pipeline.py:210`)
  - `process_asset()` (`src/rag_assistant/ingest/pipeline.py:88`)
  - `render_pdf_to_images()` (`src/rag_assistant/ingest/render/pdf_to_images.py:11`)
  - `get_ocr_engine()` (`src/rag_assistant/ingest/ocr/factory.py:15`)
  - `chunk_ocr_blocks()` (`src/rag_assistant/ingest/chunking/layout_chunker.py:23`)
  - `Embedder.embed_texts()` (`src/rag_assistant/retrieval/embedder.py:67`)
  - `QdrantStore.upsert_chunks()` (`src/rag_assistant/retrieval/vector_store/qdrant.py:148`)
- **Data persisted**
  - SQLite status + artifacts tables:
    - `asset_pages` (`src/rag_assistant/db/schema.sql:51`)
    - `asset_ocr_pages` (`src/rag_assistant/db/schema.sql:62`)
    - `chunks` (`src/rag_assistant/db/schema.sql:74`)
    - `asset_index_status` (`src/rag_assistant/db/schema.sql:44`) + extra columns added by migration logic (`src/rag_assistant/db/sqlite.py:81`)
  - Filesystem:
    - `data/subjects/<subject_id>/pages/<asset_id>/page_XXXX.png` (`src/rag_assistant/ingest/pipeline.py:100`, `src/rag_assistant/ingest/render/pdf_to_images.py:27`)
    - `data/subjects/<subject_id>/ocr/<asset_id>/page_XXXX.json` (`src/rag_assistant/ingest/pipeline.py:101`, `src/rag_assistant/ingest/ocr/paddle.py:52`)
    - `data/subjects/<subject_id>/processed/chunks/<asset_id>.jsonl` (`src/rag_assistant/ingest/pipeline.py:102`, `src/rag_assistant/ingest/chunking/layout_chunker.py:80`)
  - Qdrant:
    - Upsert payload includes `subject_id, asset_id, page_num, image_path, source, chunk_id, text, preview` (`src/rag_assistant/ingest/pipeline.py:184`)
    - Point IDs are deterministic UUIDv5 (`src/rag_assistant/vectorstore/point_id.py:10`, `src/rag_assistant/ingest/pipeline.py:183`)

**Incremental ingestion logic (what to re-run)**
- **Where**
  - Stage gating: `src/rag_assistant/ingest/pipeline.py:24`, `src/rag_assistant/ingest/pipeline.py:28`
  - Stage storage: `src/rag_assistant/services/asset_service.py:117`
- **How it works**
  - Stages ordered in `STAGE_ORDER = ["stored","rendered","ocr_done","chunked","embedded","indexed","missing","failed"]` (`src/rag_assistant/ingest/pipeline.py:24`)
  - `_should_run()` reruns when current stage is earlier, missing/unknown, or "failed" (`src/rag_assistant/ingest/pipeline.py:28`)
  - `process_subject_new_assets()` skips assets already at `indexed` (`src/rag_assistant/ingest/pipeline.py:210`)
  - Reuse behavior:
    - If not re-rendering, loads pages from `asset_pages` (`src/rag_assistant/ingest/pipeline.py:115`)
    - If not re-chunking, reads cached `processed/chunks/<asset_id>.jsonl` if present (`src/rag_assistant/ingest/pipeline.py:166`)
- **Persisted state**
  - `asset_index_status(stage,updated_at,error,ocr_engine,warning)` is upserted and mirrored into `assets.status` (`src/rag_assistant/services/asset_service.py:117`)

**Chat RAG (retrieval → context expansion → answer generation → citations)**
- **Where**
  - Service entry: `src/rag_assistant/services/chat_service.py:9`
  - Core logic: `src/rag_assistant/rag/answerer.py:56`
  - Retrieval: `src/rag_assistant/retrieval/vector_store/qdrant.py:228`
  - Context expansion: `src/rag_assistant/retrieval/context_expander.py:33`
  - LLM dispatch: `src/rag_assistant/llm/provider.py:19`
  - Prompt: `src/rag_assistant/llm/prompts/answer.md:1`
- **Key functions/classes**
  - `answerer.ask()` returns `{answer,citations,debug,...}` (`src/rag_assistant/rag/answerer.py:56`)
  - `QdrantStore.search()` includes subject filter + retry without filter (`src/rag_assistant/retrieval/vector_store/qdrant.py:228`)
  - `expand_with_neighbors()` loads neighbor chunks from SQLite (`src/rag_assistant/retrieval/context_expander.py:33`)
  - `generate_answer()` calls Ollama or OpenAI chat (`src/rag_assistant/llm/provider.py:19`)
- **Persisted data**
  - Reads SQLite `chunks` for neighbor expansion (`src/rag_assistant/retrieval/context_expander.py:28`)
  - Qdrant retrieval uses payload fields written at ingest (`src/rag_assistant/ingest/pipeline.py:184`)

**UI pages and what they do**
- Home: create/select subject, initializes config + DB (`apps/streamlit/Home.py:10`)
- Chat: send question, shows citations + retrieval debug toggle (`apps/streamlit/pages/1_Chat.py:7`, `src/rag_assistant/ui/chat_render.py:11`)
- Upload: upload files, run indexing, OCR diagnostics, missing asset cleanup (`apps/streamlit/pages/2_Upload.py:20`)
- Study Tools: placeholder buttons only (`apps/streamlit/pages/3_Study_Tools.py:8`)

**CLI commands implemented (exact invocations)**
- CLI entry: `uv run python -m rag_assistant --help` (`README.md:15`, `src/rag_assistant/__main__.py:1`)
- Subjects:
  - `uv run python -m rag_assistant subjects` (`src/rag_assistant/cli.py:46`)
  - `uv run python -m rag_assistant subjects --create "My Course"` (`README.md:31`, `src/rag_assistant/cli.py:46`)
- Reset all assets (runs bash script):
  - `uv run python -m rag_assistant reset-all` (`README.md:47`, `src/rag_assistant/cli.py:50`)
- Doctor / OCR self-test:
  - `uv run python -m rag_assistant doctor` (`src/rag_assistant/cli.py:53`, `src/rag_assistant/ingest/ocr/selftest.py:14`)
- Stubbed (prints “Not implemented yet” JSON):
  - `ingest`, `ask`, `summarize`, `flashcards`, `quiz`, `eval` (`src/rag_assistant/cli.py:39`)

**Reset / rebuild / doctor tooling**
- Reset one asset artifacts (DB rows + derived files + best-effort Qdrant delete): `scripts/reset_asset.sh:1` (documented in `README.md:35`)
- Reset all assets artifacts: `scripts/reset_all_assets.sh:1` (documented in `README.md:42`)
- Missing-asset cleanup (removes DB rows; optional Qdrant delete): `src/rag_assistant/services/cleanup_service.py:13`, exposed in UI (`apps/streamlit/pages/2_Upload.py:91`)
- OCR self-test (creates temp image + runs chosen engine): `src/rag_assistant/ingest/ocr/selftest.py:14`, exposed in UI (`apps/streamlit/pages/2_Upload.py:32`) and CLI (`src/rag_assistant/cli.py:91`)

---

## 3) Repo Structure Overview

**Top-level (key items)**
```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── config/
│   └── default.yaml
├── apps/
│   └── streamlit/
│       ├── Home.py
│       └── pages/
│           ├── 1_Chat.py
│           ├── 2_Upload.py
│           └── 3_Study_Tools.py
├── src/
│   └── rag_assistant/
│       ├── cli.py
│       ├── config.py
│       ├── db/
│       ├── ingest/
│       ├── retrieval/
│       ├── rag/
│       ├── llm/
│       ├── services/
│       └── ui/
├── scripts/
│   ├── run_qdrant.sh
│   ├── reset_asset.sh
│   └── reset_all_assets.sh
└── tests/
    └── test_*.py
```

**Purpose of major folders**
- `src/rag_assistant/` — application logic (config, DB, ingest pipeline, retrieval/RAG, UI helpers, services) (`src/rag_assistant/config.py:1`, `src/rag_assistant/ingest/pipeline.py:1`, `src/rag_assistant/rag/answerer.py:1`)
- `apps/streamlit/` — Streamlit entrypoint + pages (`apps/streamlit/Home.py:1`, `apps/streamlit/pages/2_Upload.py:1`)
- `scripts/` — dev tooling (start Qdrant, reset artifacts) (`scripts/run_qdrant.sh:1`, `scripts/reset_asset.sh:1`)
- `data/` — runtime data root (uploads, derived artifacts, SQLite db) configured by `app.data_root` (`config/default.yaml:4`, `src/rag_assistant/config.py:148`)
- `tests/` — pytest suite for config/db/pipeline/retrieval/UI helpers (`tests/test_imports.py:1`)

---

## 4) Configuration & Environment

**How config is loaded**
- `load_config()` reads YAML (`config/default.yaml:1` by default) + applies env overrides + loads `.env` via `dotenv.load_dotenv()` (`src/rag_assistant/config.py:148`)
- It also creates `data_root`, `logs_dir`, and DB parent directory on load (`src/rag_assistant/config.py:160`)

**Key config fields (YAML)**
- `app`: name/environment/data_root/logs_dir (`config/default.yaml:1`)
- `database.sqlite_path` (`config/default.yaml:7`)
- `qdrant.url`, `qdrant.collection`, `qdrant.vector_size` (`config/default.yaml:10`)
- `ingest.ocr_engine`, `ingest.pdf_dpi`, chunk sizing, tesseract paths (`config/default.yaml:21`)
- `retrieval.top_k`, neighbor expansion knobs, `min_score` (`config/default.yaml:31`)
- `llm.provider`, `llm.model`, `llm.base_url` (`config/default.yaml:37`)
- `embeddings.provider`, `embeddings.model`, `embeddings.vector_size` (`config/default.yaml:46`)

**Env var overrides (subset)**
- Paths/logging: `DATA_ROOT`, `LOGS_DIR`, `DB_PATH`, `LOG_LEVEL` (`src/rag_assistant/config.py:57`)
- Qdrant: `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_VECTOR_SIZE` (`src/rag_assistant/config.py:65`)
- Retrieval: `RETRIEVAL_TOP_K`, `RETRIEVAL_NEIGHBOR_WINDOW`, `RETRIEVAL_MAX_NEIGHBOR_CHUNKS`, `RETRIEVAL_MIN_SCORE` (`src/rag_assistant/config.py:69`)
- LLM: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_S` (`src/rag_assistant/config.py:70`)
- Embeddings: `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL`, `EMBEDDINGS_VECTOR_SIZE` (`src/rag_assistant/config.py:75`)
- OCR: `OCR_ENGINE`, `TESSERACT_CMD`, `TESSDATA_DIR` (`src/rag_assistant/config.py:78`)
- Example `.env` template: `.env.example:1`

**Default providers currently configured**
- OCR: `tesseract` (`config/default.yaml:22`)
- LLM: `ollama` (`config/default.yaml:38`)
- Embeddings: `local` E5-small (`config/default.yaml:46`)
- Vector store: Qdrant collection `rag_chunks_e5` at `http://localhost:6333` (`config/default.yaml:14`)

**Commands to run locally**
- Install:
  - `uv venv --python 3.11` (`README.md:7`) or `make venv` (`Makefile:5`)
  - `uv pip install -e ".[dev]"` (`README.md:11`) or `make install` (`Makefile:8`)
- Run UI:
  - `uv run streamlit run apps/streamlit/Home.py` (`README.md:19`) or `make ui` (`Makefile:14`)
- Start Qdrant (Docker):
  - `make qdrant` (`README.md:51`, `Makefile:20`, `scripts/run_qdrant.sh:1`)
- Start Ollama:
  - Code’s error message suggests `ollama serve` (`src/rag_assistant/llm/ollama_client.py:35`)
- Run indexing:
  - Via UI button “Index new uploads” (`apps/streamlit/pages/2_Upload.py:48`)
  - CLI ingest command is **Not implemented** (`src/rag_assistant/cli.py:39`)
- Run tests:
  - `uv run pytest -q` (`README.md:23`) or `make test` (`Makefile:11`)

---

## 5) Data Model

### SQLite schema (tables + columns)
Defined in `src/rag_assistant/db/schema.sql:1` (plus runtime migration additions in `src/rag_assistant/db/sqlite.py:81`).

- `subjects(subject_id PK, name, created_at, meta_json)` (`src/rag_assistant/db/schema.sql:23`)
- `assets(asset_id PK, subject_id FK, original_filename, stored_path, sha256, size_bytes, mime_type, created_at, status, meta_json)` (`src/rag_assistant/db/schema.sql:30`)
- `asset_index_status(asset_id PK, stage, updated_at, error)` (`src/rag_assistant/db/schema.sql:44`)
  - Migration logic may add: `ocr_engine`, `warning` (`src/rag_assistant/db/sqlite.py:81`)
- `asset_pages(id PK, asset_id, page_num, image_path, width, height, created_at, UNIQUE(asset_id,page_num))` (`src/rag_assistant/db/schema.sql:51`)
- `asset_ocr_pages(id PK, asset_id, page_num, ocr_json_path, text_len, avg_conf, needs_caption, created_at, UNIQUE(asset_id,page_num))` (`src/rag_assistant/db/schema.sql:62`)
- `chunks(chunk_id PK, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at)` (`src/rag_assistant/db/schema.sql:74`)
- Also present but **not referenced in `src/`** (only defined in schema):
  - `documents` (`src/rag_assistant/db/schema.sql:1`)
  - `runs` (`src/rag_assistant/db/schema.sql:8`)
  - `quiz_attempts` (`src/rag_assistant/db/schema.sql:16`)
- Migration bookkeeping:
  - `schema_migrations(id PK, applied_at)` created at runtime (`src/rag_assistant/db/sqlite.py:90`)

### Relations (logical)
- `subjects.subject_id → assets.subject_id` (enforced FK) (`src/rag_assistant/db/schema.sql:41`)
- `assets.asset_id → asset_pages.asset_id / asset_ocr_pages.asset_id / chunks.asset_id` (not declared as FK, but used throughout) (`src/rag_assistant/ingest/pipeline.py:43`, `src/rag_assistant/ingest/pipeline.py:52`, `src/rag_assistant/ingest/pipeline.py:68`)

### Migrations (what exists + how applied)
- SQL files: `src/rag_assistant/db/migrations/001_init.sql:1`, `src/rag_assistant/db/migrations/003_ingest_tables.sql:1`, placeholders `004_*.sql:1`, `005_add_chunk_id_to_chunks.sql:1`
- Applied by `init_db()` calling `apply_migrations()` (`src/rag_assistant/db/sqlite.py:15`, `src/rag_assistant/db/sqlite.py:90`)
- Some migration behavior is **implemented in Python**:
  - `ensure_chunks_columns()` + chunk_id backfill (`src/rag_assistant/db/sqlite.py:46`)
  - `ensure_asset_status_columns()` (`src/rag_assistant/db/sqlite.py:81`)

### Qdrant data model
- **Collection name**: uses `cfg.qdrant.collection` (not `collection_name`) (`config/default.yaml:15`, `src/rag_assistant/retrieval/vector_store/qdrant.py:106`)
- **Vector size**: uses `cfg.embeddings.vector_size` and enforces match with existing collection (`config/default.yaml:49`, `src/rag_assistant/retrieval/vector_store/qdrant.py:125`)
- **Distance**: `COSINE` (`src/rag_assistant/retrieval/vector_store/qdrant.py:143`)
- **Payload fields written** (ingest): `subject_id, asset_id, page_num, image_path, source, chunk_id, text, preview` (`src/rag_assistant/ingest/pipeline.py:184`)
- **Point ID strategy**: deterministic UUIDv5 from identity string (`src/rag_assistant/vectorstore/point_id.py:10`, `src/rag_assistant/ingest/pipeline.py:183`)
  - Rationale (from code behavior): stable IDs allow idempotent upserts per chunk identity.

---

## 6) Ingestion Pipeline Deep Dive

**Step-by-step indexing**
1) **Input asset lookup + stage gating**
- Uses `asset_index_status.stage` (or `assets.status`) and `_should_run()` vs `STAGE_ORDER` (`src/rag_assistant/ingest/pipeline.py:24`, `src/rag_assistant/ingest/pipeline.py:28`, `src/rag_assistant/ingest/pipeline.py:91`)
- Missing raw file → mark `missing` and stop (`src/rag_assistant/ingest/pipeline.py:94`)

2) **Render pages**
- PDFs: `render_pdf_to_images()` writes `page_XXXX.png` using PyMuPDF at `ingest.pdf_dpi` (`src/rag_assistant/ingest/pipeline.py:106`, `src/rag_assistant/ingest/render/pdf_to_images.py:11`)
- Images: `normalize_image_to_page()` copies to `page_0001.png` and reads dims via OpenCV (`src/rag_assistant/ingest/pipeline.py:109`, `src/rag_assistant/ingest/render/image_to_page.py:12`)
- Records inserted into `asset_pages` (`src/rag_assistant/ingest/pipeline.py:43`)

3) **OCR engine selection + run**
- Select engine via `get_ocr_engine()` based on `ingest.ocr_engine`:
  - explicit `tesseract` or `paddle`, else `auto` tries Paddle→Tesseract→Stub (`src/rag_assistant/ingest/ocr/factory.py:15`)
- Per page: `ocr_engine.ocr_page(image_path,page_num)` (`src/rag_assistant/ingest/pipeline.py:123`)
- Output normalized to `{page, blocks:[{text,bbox,confidence}], width,height}` (`src/rag_assistant/ingest/ocr/normalize.py:12`)
- Per page artifact:
  - JSON saved to `data/subjects/<subject>/ocr/<asset_id>/page_XXXX.json` (`src/rag_assistant/ingest/pipeline.py:136`, `src/rag_assistant/ingest/ocr/paddle.py:52`)
  - Stats (text_len/avg_conf/needs_caption) computed (`src/rag_assistant/ingest/ocr/paddle.py:59`) and stored in `asset_ocr_pages` (`src/rag_assistant/ingest/pipeline.py:52`)
- Stage update includes engine metadata and warnings (`src/rag_assistant/ingest/pipeline.py:141`, `src/rag_assistant/services/asset_service.py:117`)

4) **Chunking**
- Reads saved OCR JSONs from `asset_ocr_pages.ocr_json_path` (`src/rag_assistant/ingest/pipeline.py:147`)
- `chunk_ocr_blocks()`:
  - sorts blocks by reading order (y then x), unions bbox, builds chunks by `max_chunk_chars/min_chunk_chars` with `overlap_blocks` (`src/rag_assistant/ingest/chunking/layout_chunker.py:23`)
  - chunk_id is `sha256(f"{asset_id}:{page_num}:{start_block}:{end_block}")[:20]` (`src/rag_assistant/ingest/chunking/layout_chunker.py:61`)
- Writes cache JSONL: `processed/chunks/<asset_id>.jsonl` (`src/rag_assistant/ingest/pipeline.py:161`, `src/rag_assistant/ingest/chunking/layout_chunker.py:80`)
- Upserts into SQLite `chunks` (`src/rag_assistant/ingest/pipeline.py:68`)

5) **Embedding generation**
- `Embedder.embed_texts()`:
  - local: sentence-transformers `encode(..., normalize_embeddings=True)` (`src/rag_assistant/retrieval/embedder.py:53`)
  - openai: `client.embeddings.create(model=cfg.llm.embed_model, input=texts)` (`src/rag_assistant/retrieval/embedder.py:61`)
- Pipeline embeds all chunk texts for the asset in one call (`src/rag_assistant/ingest/pipeline.py:175`)

6) **Qdrant upsert**
- Deterministic point id per chunk: `make_point_uuid(identity)` (`src/rag_assistant/ingest/pipeline.py:183`, `src/rag_assistant/vectorstore/point_id.py:10`)
- `QdrantStore.ensure_collection()` creates collection if missing, and checks vector-size compatibility if it exists (`src/rag_assistant/retrieval/vector_store/qdrant.py:125`)
- Upsert uses `qmodels.Batch(ids,vectors,payloads)` (`src/rag_assistant/retrieval/vector_store/qdrant.py:148`)

7) **Failure handling + status**
- Any exception sets stage `failed` with error string and re-raises (`src/rag_assistant/ingest/pipeline.py:205`)
- `process_subject_new_assets()` continues on per-asset failures and returns summary (`src/rag_assistant/ingest/pipeline.py:210`)

**Artifacts under `data/subjects/<subject_id>/...`**
- Raw uploads: `raw/` (`src/rag_assistant/services/subject_service.py:57`)
- Rendered pages: `pages/<asset_id>/page_XXXX.png` (`src/rag_assistant/ingest/pipeline.py:100`)
- OCR JSON: `ocr/<asset_id>/page_XXXX.json` (`src/rag_assistant/ingest/pipeline.py:101`)
- Chunk cache: `processed/chunks/<asset_id>.jsonl` (`src/rag_assistant/ingest/pipeline.py:102`)

**Performance notes (from implementation)**
- Most expensive steps are likely **PDF rasterization** (DPI-scaled pixmaps) (`src/rag_assistant/ingest/render/pdf_to_images.py:22`), **OCR** (Paddle/Tesseract) (`src/rag_assistant/ingest/ocr/paddle.py:22`, `src/rag_assistant/ingest/ocr/tesseract.py:37`), and **embedding** (torch-backed sentence-transformers) (`pyproject.toml:21`, `src/rag_assistant/retrieval/embedder.py:53`).

---

## 7) Retrieval + Answering Deep Dive

**Query embedding computed + normalized**
- `answerer.ask()` embeds `[question]` via `Embedder.embed_texts()` (`src/rag_assistant/rag/answerer.py:73`)
- Converts embedding to `list[float]` (supports `.tolist()`) and validates dimension + NaN/inf (`src/rag_assistant/rag/answerer.py:80`, `src/rag_assistant/rag/answerer.py:44`)

**Qdrant search compatibility logic (client API differences)**
- Wrapper `search_points()`:
  - prefers `client.search(..., query_vector=..., query_filter=...)`
  - falls back to `filter=` on TypeError
  - supports `client.query_points()` and switches between `query_vector` vs `query`, and `query_filter` vs `filter` based on signature (`src/rag_assistant/retrieval/vector_store/qdrant.py:64`)

**Subject filtering + retry behavior**
- `QdrantStore.search()` builds a Qdrant filter on `subject_id` payload (`src/rag_assistant/retrieval/vector_store/qdrant.py:239`)
- If no hits and subject was provided, retries without filter (`src/rag_assistant/retrieval/vector_store/qdrant.py:275`)
- `answerer.ask()` also retries by calling `store.search(..., subject_id=None)` if its first call returns no hits (`src/rag_assistant/rag/answerer.py:85`)

**Score thresholds (`min_score`)**
- Filters hits with `score >= retrieval.min_score`; if that removes everything, falls back to original hits (`src/rag_assistant/rag/answerer.py:108`, `config/default.yaml:35`)

**Hydration from SQLite**
- If a Qdrant hit payload lacks `text`, `_hydrate_payload()` fetches `chunks` row by `chunk_id` and fills `text,page_num,asset_id,subject_id,bbox_json,start_block,end_block` (`src/rag_assistant/retrieval/vector_store/qdrant.py:156`)

**Neighbor-page/context expansion**
- `expand_with_neighbors()`:
  - groups hits by `asset_id`, collects neighbor pages within `window`, queries SQLite `chunks` for those pages, adds up to `max_extra` unique chunks (`src/rag_assistant/retrieval/context_expander.py:33`)
- Called from answerer with `neighbor_window` + `max_neighbor_chunks` (`src/rag_assistant/rag/answerer.py:114`, `config/default.yaml:33`)

**Prompt template**
- Loaded from `src/rag_assistant/llm/prompts/answer.md:1` and rendered via Jinja2 with `context` + `question` (`src/rag_assistant/rag/answerer.py:19`, `src/rag_assistant/rag/answerer.py:125`)

**Citation construction**
- Returns citations for each chunk in the final context set:
  - `asset_id`, `filename`, `page`, `chunk_id`, `quote` (first 240 chars), `image_path` (`src/rag_assistant/rag/answerer.py:140`)

**Retrieval debug info**
- Structure: `RetrievalDebug` dataclass (`src/rag_assistant/retrieval/debug.py:8`)
- Returned on success and “not found” paths (`src/rag_assistant/rag/answerer.py:158`, `src/rag_assistant/rag/answerer.py:91`)
- Displayed in UI: “Show retrieval debug” checkbox shows stored `last_retrieval_debug` JSON (`src/rag_assistant/ui/chat_render.py:38`)

---

## 8) Streamlit UI Deep Dive

**Home**
- Entry: `apps/streamlit/Home.py:1`
- Purpose: initializes config + DB, creates subjects, selects active subject
- Key widgets/state:
  - `st.form` for create subject (`apps/streamlit/Home.py:30`)
  - `st.selectbox` to choose subject; stored in session state (`apps/streamlit/Home.py:47`, `src/rag_assistant/ui/session_state.py:34`)
  - Sidebar subject chooser + nav links (`apps/streamlit/Home.py:27`, `src/rag_assistant/ui/sidebar.py:29`)

**Upload**
- Entry: `apps/streamlit/pages/2_Upload.py:1`
- Purpose: upload files, run indexing, show asset statuses, provide OCR diagnostics and missing-asset cleanup
- Key widgets/state:
  - OCR debug expander: shows resolved tesseract paths and “Run OCR self-test” (`apps/streamlit/pages/2_Upload.py:20`, `src/rag_assistant/ingest/ocr/selftest.py:14`)
  - `st.file_uploader` + “Save files” button writes assets (`apps/streamlit/pages/2_Upload.py:39`, `src/rag_assistant/services/asset_service.py:55`)
  - “Index new uploads” button calls pipeline (`apps/streamlit/pages/2_Upload.py:48`, `src/rag_assistant/ingest/pipeline.py:210`)
  - Missing asset cleanup calls `cleanup_service.remove_assets(..., remove_vectors=False)` (`apps/streamlit/pages/2_Upload.py:97`, `src/rag_assistant/services/cleanup_service.py:13`)
- Status display:
  - Uses `assets.status` and `asset_index_status.stage/error/ocr_engine` (`apps/streamlit/pages/2_Upload.py:65`, `src/rag_assistant/services/asset_service.py:117`)

**Chat**
- Entry: `apps/streamlit/pages/1_Chat.py:1`
- Purpose: ask questions, show answer, citations, debug
- Key widgets/state:
  - `st.text_input` question + “Send” button (`src/rag_assistant/ui/chat_render.py:16`)
  - Stores messages in `st.session_state["chat_messages"]` (`src/rag_assistant/ui/session_state.py:8`)
  - Stores last retrieval debug and context expansion count (`src/rag_assistant/ui/chat_render.py:18`)
- Citations rendering:
  - Lists citations and optionally shows page image if the file exists (`src/rag_assistant/ui/citations_render.py:10`)

**Study Tools**
- Entry: `apps/streamlit/pages/3_Study_Tools.py:1`
- Purpose: placeholder buttons (summary/flashcards/quiz) — not implemented (`apps/streamlit/pages/3_Study_Tools.py:15`)

---

## 9) Testing & Quality

**Tests present (files)**
- `tests/test_answerer_embedding_normalization.py:1`
- `tests/test_answerer_flat_hits.py:1`
- `tests/test_chat_empty_index_message.py:1`
- `tests/test_chunk_id_backfill_format.py:1`
- `tests/test_chunk_migration.py:1`
- `tests/test_chunker.py:1`
- `tests/test_config_load.py:1`
- `tests/test_context_expander.py:1`
- `tests/test_db_init.py:1`
- `tests/test_db_migration_chunks_asset_id.py:1`
- `tests/test_embedding_validator.py:1`
- `tests/test_imports.py:1`
- `tests/test_ingest_tables.py:1`
- `tests/test_local_embedder_dim.py:1`
- `tests/test_missing_assets_cleanup.py:1`
- `tests/test_ocr_normalization.py:1`
- `tests/test_ollama_client.py:1`
- `tests/test_paddle_ocr_api_guard.py:1`
- `tests/test_paddle_wrapper_kwargs.py:1`
- `tests/test_pdf_render.py:1`
- `tests/test_pipeline_progression.py:1`
- `tests/test_qdrant_collection_count_compat.py:1`
- `tests/test_qdrant_hydration.py:1`
- `tests/test_qdrant_payload_preserved.py:1`
- `tests/test_qdrant_point_ids_uuid.py:1`
- `tests/test_qdrant_retry_and_payload.py:1`
- `tests/test_qdrant_search_compat.py:1`
- `tests/test_qdrant_search_vector_param.py:1`
- `tests/test_retrieval_retry_no_filter.py:1`
- `tests/test_selftest_uses_file.py:1`
- `tests/test_sidebar_nav_paths.py:1`
- `tests/test_subjects_assets.py:1`
- `tests/test_tesseract_helpers.py:1`
- `tests/test_tesseract_resolution.py:1`

**What they cover (high level)**
- Config + env overrides + directory creation (`tests/test_config_load.py:1`)
- SQLite init + migrations/backfill behavior (`tests/test_db_init.py:1`, `tests/test_chunk_id_backfill_format.py:1`)
- OCR normalization + Paddle/Tesseract selection guards (`tests/test_ocr_normalization.py:1`, `tests/test_paddle_wrapper_kwargs.py:1`)
- Rendering PDF to images (`tests/test_pdf_render.py:1`)
- Pipeline stage progression with mocked OCR/embeddings/Qdrant (`tests/test_pipeline_progression.py:1`)
- Qdrant API compatibility + hydration + retry behavior (`tests/test_qdrant_search_vector_param.py:1`, `tests/test_qdrant_hydration.py:1`)
- Answerer behavior (embedding normalization, retries, citation building) (`tests/test_answerer_flat_hits.py:1`)
- Streamlit sidebar path safety (`tests/test_sidebar_nav_paths.py:1`)

**How to run**
- `uv run pytest -q` (`README.md:23`) or `make test` (`Makefile:11`)

**Known flaky/blocked issues (from test implementation)**
- `tests/test_qdrant_collection_count_compat.py:1` constructs a real `QdrantStore()` before monkeypatching its client, so it can require a reachable Qdrant at `config/default.yaml:14` during test collection/runtime (`src/rag_assistant/retrieval/vector_store/qdrant.py:101`).
- Tesseract resolution tests may depend on local availability of tessdata/traineddata since `_resolve_tessdata_dir` checks filesystem candidates (`src/rag_assistant/ingest/ocr/tesseract.py:61`, `tests/test_tesseract_resolution.py:1`).

**Warnings currently seen**
- Not executed in this report (no runtime logs captured). The code does surface runtime warnings/errors into `asset_index_status.warning` and `asset_index_status.error` during OCR selection and failures (`src/rag_assistant/ingest/ocr/factory.py:33`, `src/rag_assistant/services/asset_service.py:117`).

---

## 10) Known Issues / Gaps / TODOs (Prioritized)

1) **README vs default config mismatch**
- README says indexing “runs PaddleOCR” and chat requires `OPENAI_API_KEY` (`README.md:32`), but default config selects `tesseract` OCR and `ollama` LLM + `local` embeddings (`config/default.yaml:22`, `config/default.yaml:38`, `config/default.yaml:46`). Suggested fix: update `README.md:28` to reflect current defaults and optional providers.

2) **Qdrant config duplication / unused fields**
- Both `qdrant.collection_name` and `qdrant.collection` exist, but code uses `cfg.qdrant.collection` (`config/default.yaml:13`, `config/default.yaml:15`, `src/rag_assistant/retrieval/vector_store/qdrant.py:106`). Suggested fix: remove or consistently use one field; clarify `host/port` vs `url`.

3) **CLI indexing + “ask” are stubs**
- `ingest`/`ask` commands are defined but not implemented (`src/rag_assistant/cli.py:39`). Indexing is effectively UI-only (`apps/streamlit/pages/2_Upload.py:48`). Suggested fix: wire CLI `ingest` to `process_subject_new_assets()` (`src/rag_assistant/ingest/pipeline.py:210`) and CLI `ask` to `answerer.ask()` (`src/rag_assistant/rag/answerer.py:56`).

4) **Test suite may require a running Qdrant**
- `tests/test_qdrant_collection_count_compat.py:1` instantiates `QdrantStore()` (which calls `ensure_collection()`) before monkeypatching. Suggested fix: monkeypatch `QdrantClient`/`load_config` before construction, or lazily connect inside `QdrantStore`.

5) **Tesseract OCR loses layout; chunking becomes coarse**
- Tesseract engine uses `pytesseract.image_to_string()` and normalizes to a single block with default bbox (`src/rag_assistant/ingest/ocr/tesseract.py:52`, `src/rag_assistant/ingest/ocr/normalize.py:26`). That limits layout-aware chunking (`src/rag_assistant/ingest/chunking/layout_chunker.py:23`). Suggested fix: use Tesseract TSV/HOCR to retain boxes, or default to PaddleOCR when available.

6) **Asset deletion/replace flows are minimal**
- Only “missing asset cleanup” exists, and reset scripts are dev tooling (`src/rag_assistant/services/cleanup_service.py:13`, `scripts/reset_asset.sh:1`). Suggested fix: add UI + service for deleting/replacing assets and optionally removing vectors.

7) **Embedding provider vs vector size needs validation**
- `Embedder` can switch to OpenAI embeddings (`src/rag_assistant/retrieval/embedder.py:47`), but vector dimension must match `embeddings.vector_size` and Qdrant collection size (`src/rag_assistant/rag/answerer.py:84`, `src/rag_assistant/retrieval/vector_store/qdrant.py:137`). Suggested fix: validate config at startup and/or auto-select correct `vector_size` for chosen model.

8) **Status schema semantics are mixed**
- OCR engine message is written into `asset_index_status.error` even when not an error (`src/rag_assistant/ingest/pipeline.py:141`, `src/rag_assistant/services/asset_service.py:117`). Suggested fix: separate `message` vs `error`, or use `warning` consistently.

9) **Repo contains committed runtime data**
- `data/` includes a SQLite db and example subject artifacts in this working copy (not found in code, but present in repo tree). Suggested fix: add `data/` to `.gitignore` or move example data under an explicit `examples/` directory. (If this is intentional, document it in `README.md:57`.)

---

## 11) “How to Use It Today” Quickstart (macOS/Linux)

1) Create env + install:
- `uv venv --python 3.11` (`README.md:7`) or `make venv` (`Makefile:5`)
- `uv pip install -e ".[dev]"` (`README.md:11`) or `make install` (`Makefile:8`)

2) Start Qdrant (Docker required):
- `make qdrant` (`README.md:51`, `scripts/run_qdrant.sh:1`)

3) Start Ollama (for default `llm.provider=ollama`):
- `ollama serve` (suggested by error message in `src/rag_assistant/llm/ollama_client.py:35`)
- Ensure the configured model exists in Ollama (`config/default.yaml:39`). Command to pull the model is **Not found in repo**.

4) Run Streamlit UI:
- `make ui` (`Makefile:14`) or `uv run streamlit run apps/streamlit/Home.py` (`README.md:19`)

5) Use the app:
- Home: create a subject (`apps/streamlit/Home.py:29`)
- Upload: choose a PDF/image, click “Save files”, then “Index new uploads” (`apps/streamlit/pages/2_Upload.py:39`, `apps/streamlit/pages/2_Upload.py:48`)
- Chat: ask a question; inspect citations and “Show retrieval debug” if needed (`apps/streamlit/pages/1_Chat.py:7`, `src/rag_assistant/ui/chat_render.py:38`)

6) Helpful diagnostics:
- OCR self-test in UI or CLI:
  - UI button (`apps/streamlit/pages/2_Upload.py:32`)
  - `uv run python -m rag_assistant doctor` (`src/rag_assistant/cli.py:53`)
- Reset derived artifacts:
  - `./scripts/reset_asset.sh <asset_id>` (`README.md:35`, `scripts/reset_asset.sh:1`)
  - `./scripts/reset_all_assets.sh` or `uv run python -m rag_assistant reset-all` (`README.md:42`, `src/rag_assistant/cli.py:50`)
