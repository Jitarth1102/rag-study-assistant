# Project State

## End-to-end behavior
- Users create/select a subject in Streamlit Home, then upload files under that subject (`apps/streamlit/Home.py`, `apps/streamlit/pages/2_Upload.py`).
- Upload saves files to `data/subjects/<subject>/raw/` and registers metadata in SQLite (`src/rag_assistant/services/asset_service.py` lines ~19-98).
- Ingestion pipeline renders pages → OCR → chunk → embed → upsert to Qdrant with deterministic point IDs; stages mirrored in `asset_index_status`/`assets.status` (`src/rag_assistant/ingest/pipeline.py` lines ~25-206).
- Chat page embeds the question, searches Qdrant (slides + notes), expands neighbor chunks from SQLite, calls LLM to answer with citations (`apps/streamlit/pages/1_Chat.py`; `src/rag_assistant/ui/chat_render.py` lines ~11-93; `src/rag_assistant/services/chat_service.py` lines ~10-37; `src/rag_assistant/rag/answerer.py` lines ~12-226).
- Notes page lets users generate Markdown notes per asset via LLM (optional web), edit, and reindex notes chunks into Qdrant; latest notes are rendered in the UI (`apps/streamlit/pages/4_Notes.py`).

## Key structure
- `apps/streamlit/` Streamlit entrypoints and pages; shared UI helpers in `src/rag_assistant/ui/`.
- `src/rag_assistant/ingest/` rendering, OCR, chunking, and pipeline orchestration.
- `src/rag_assistant/rag/` answerer + judge prompts; `src/rag_assistant/retrieval/` embedding, neighbor expansion, Qdrant client.
- `src/rag_assistant/services/` subject/asset lifecycle, chat, health, notes, cleanup.
- `config/default.yaml` defaults; `src/rag_assistant/config.py` loads YAML + env overrides and ensures data/log/db directories.
- `src/rag_assistant/db/` schema, migrations, SQLite helpers.
- `scripts/` operational helpers (reset, qdrant).

## Database schema & migrations
- Core tables: `subjects`, `assets`, `asset_index_status`, `asset_pages`, `asset_ocr_pages`, `chunks`, `runs`, `documents`, `quiz_attempts` (`src/rag_assistant/db/schema.sql` lines ~1-52).
- Notes tables: `notes` (notes_id PK, subject_id, asset_id, version, markdown, generated_by, timestamps, meta_json) and `notes_chunks` (notes_chunk_id PK, notes_id, subject_id, asset_id, section_title, text, created_at) with unique (subject_id, asset_id, version) (`src/rag_assistant/db/schema.sql` lines ~54-80).
- Migrations live under `src/rag_assistant/db/migrations/` with runner `apply_migrations` creating `schema_migrations` and special handling for legacy chunk columns (`src/rag_assistant/db/sqlite.py` lines ~15-120).
- Asset cleanup also deletes notes and note chunks, plus asset-related rows (`src/rag_assistant/db/sqlite.py` lines ~122-140).

## Ingestion and storage
- Pipeline stages: render PDF/image to pages, OCR each page, chunk OCR blocks with overlap-aware layout chunker, write chunks JSONL, upsert `chunks` table, embed texts, upsert Qdrant payloads (`src/rag_assistant/ingest/pipeline.py` lines ~25-206).
- Chunk IDs derived from asset/page/block ranges; JSONL stored under `data/subjects/<subject>/processed/chunks/` (`src/rag_assistant/ingest/chunking/layout_chunker.py` lines ~8-73).
- Embedding uses local sentence-transformers by default (`src/rag_assistant/retrieval/embedder.py` lines ~12-71); vector size from config.

## Qdrant usage & retrieval
- Config: `qdrant.collection` default `rag_chunks_e5`, url `http://localhost:6333`, vector_size 384 (`config/default.yaml`; `src/rag_assistant/config.py` lines ~23-70).
- Upsert payloads: slides include `source_type: "slide"`, subject_id, asset_id, page_num, image_path, source filename, chunk_id, text, preview; point IDs via uuid5 of subject/asset/page/chunk (`src/rag_assistant/ingest/pipeline.py` lines ~148-172; `src/rag_assistant/vectorstore/point_id.py`).
- Notes payloads include `source_type: "notes"`, `notes_id`, `notes_chunk_id` (also as chunk_id), section_title, text, preview, subject_id, asset_id, source label (`src/rag_assistant/services/notes_service.py` lines ~137-189).
- Retrieval: `QdrantStore.search` filters by subject_id, retries without filter if empty; hydrates payload text/page from SQLite if missing; `search_notes` filters `source_type="notes"` with similar retry (`src/rag_assistant/retrieval/vector_store/qdrant.py` lines ~20-185).
- Answerer merges slide and notes hits, neighbor-expands slides from SQLite, dedupes by chunk_id, caps context, and returns citations by type: `slide` with page/image; `notes` with notes_id/section; `web` passthrough (`src/rag_assistant/rag/answerer.py` lines ~40-206).
- Citations rendered with separate sections for slides, notes, and web (`src/rag_assistant/ui/citations_render.py` lines ~1-38).

## Web search fallback
- Judge decides web usage based on hit count/score thresholds, definitional questions, or force flag (`src/rag_assistant/rag/judge.py` lines ~7-70); config gates include `web.enabled`, min hits/score, `force_even_if_rag_strong`, domain allow/block lists (`config/default.yaml`; `src/rag_assistant/config.py` lines ~72-111).
- Search client supports SerpAPI only; enforces API key and filters results by allow/block lists (`src/rag_assistant/web/search_client.py` lines ~10-93).
- Answerer respects `max_web_queries_per_question`, dedupes URLs, caps web context to ~1200 chars, and records debug data; Streamlit sidebar lets users toggle web use, max queries, and domain lists (`src/rag_assistant/ui/chat_render.py` lines ~15-81).
- Prompts: base answer and answer_with_web templates under `src/rag_assistant/llm/prompts/` are used by answerer; notes generation uses inline Jinja template in `notes_service`.

## Notes generation/edit flow
- `notes_service.generate_notes_for_asset` loads slide chunks, optionally searches web once per judge decision (bounded by max_web_queries_per_question), prompts LLM for Markdown, stores/versions the same notes_id, chunks markdown by headings/char limit, reindexes note chunks, and returns metadata (`src/rag_assistant/services/notes_service.py` lines ~82-195).
- `update_notes` increments version, rebuilds chunks, deletes old vectors by notes_id, and re-upserts (`src/rag_assistant/services/notes_service.py` lines ~197-235).
- Streamlit Notes page drives generate/edit/save UI with error handling; PDF download is a stub (`apps/streamlit/pages/4_Notes.py`).

## Reset/cleanup
- `cleanup_service` can remove missing assets and optionally Qdrant vectors (`src/rag_assistant/services/cleanup_service.py`).
- `scripts/reset_asset.sh` deletes asset-related DB rows (including notes), derived files, and best-effort Qdrant deletion by asset_id; `scripts/reset_all_assets.sh` iterates assets (`scripts/reset_asset.sh`, `scripts/reset_all_assets.sh`).

## Tests and how to run
- Extensive pytest suite covering config, ingestion, OCR, retrieval, web guardrails, Qdrant hydration, UI helpers, etc. (`tests/`).
- Notes-specific tests: `tests/test_notes_service.py` (generation/update/web bound), `tests/test_answerer_notes_retrieval.py` (merging notes hits).
- Run via `UV_CACHE_DIR=.uv_cache uv run pytest -q` (Makefile target `make test`); note uv runtime must be available, otherwise install deps and use `pytest -q`.
