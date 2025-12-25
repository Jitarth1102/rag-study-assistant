"""Ingestion pipeline orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import logging

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, ensure_chunks_columns
from rag_assistant.ingest.chunking.layout_chunker import chunk_ocr_blocks, write_chunks_jsonl
from rag_assistant.ingest.ocr.paddle import analyze_ocr_stats, save_ocr_json
from rag_assistant.ingest.ocr.factory import get_ocr_engine
from rag_assistant.ingest.render.image_to_page import normalize_image_to_page
from rag_assistant.ingest.render.pdf_to_images import render_pdf_to_images
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore
from rag_assistant.services import asset_service
from rag_assistant.vectorstore.point_id import make_point_uuid

STAGE_ORDER = ["stored", "rendered", "ocr_done", "chunked", "embedded", "indexed", "missing", "failed"]
logger = logging.getLogger(__name__)


def _should_run(current_stage: str | None, target_stage: str) -> bool:
    if current_stage is None:
        return True
    if current_stage == "failed":
        return True
    try:
        return STAGE_ORDER.index(current_stage) < STAGE_ORDER.index(target_stage)
    except ValueError:
        return True


def _set_stage(asset_id: str, stage: str, error: str | None = None, ocr_engine: str | None = None, warning: str | None = None) -> None:
    asset_service.upsert_index_status(asset_id, stage, error, ocr_engine=ocr_engine, warning=warning)


def _insert_page_records(asset_id: str, pages: List[Dict]) -> None:
    for page in pages:
        execute(
            asset_service.get_db_path(),
            "INSERT OR REPLACE INTO asset_pages (asset_id, page_num, image_path, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?);",
            (asset_id, page["page_num"], page["image_path"], page.get("width"), page.get("height"), time.time()),
        )


def _insert_ocr_record(asset_id: str, page_num: int, ocr_path: Path, stats: dict) -> None:
    execute(
        asset_service.get_db_path(),
        "INSERT OR REPLACE INTO asset_ocr_pages (asset_id, page_num, ocr_json_path, text_len, avg_conf, needs_caption, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
        (
            asset_id,
            page_num,
            str(ocr_path),
            stats.get("text_len", 0),
            stats.get("avg_conf", 0.0),
            stats.get("needs_caption", 0),
            time.time(),
        ),
    )


def _upsert_chunks(chunks: List[dict]) -> None:
    ensure_chunks_columns(asset_service.get_db_path())
    for chunk in chunks:
        execute(
            asset_service.get_db_path(),
            "INSERT OR REPLACE INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (
                chunk["chunk_id"],
                chunk["subject_id"],
                chunk["asset_id"],
                chunk["page_num"],
                chunk["text"],
                chunk.get("bbox_json"),
                chunk["start_block"],
                chunk["end_block"],
                time.time(),
            ),
        )


def process_asset(subject_id: str, asset: dict, config=None, *, force: bool = False) -> None:
    cfg = config or load_config()
    asset_id = asset["asset_id"]
    status = asset_service.get_index_status(asset_id)
    current_stage = None if force else (status["stage"] if status else None)

    if not Path(asset["stored_path"]).exists():
        _set_stage(asset_id, "missing", error=f"raw file missing on disk: {asset['stored_path']}")
        return

    data_root = Path(cfg.app.data_root)
    asset_base = data_root / "subjects" / subject_id
    pages_dir = asset_base / "pages" / asset_id
    ocr_dir = asset_base / "ocr" / asset_id
    chunks_dir = asset_base / "processed" / "chunks"

    try:
        if _should_run(current_stage, "rendered"):
            if asset.get("mime_type", "").lower().startswith("application/pdf") or asset["stored_path"].lower().endswith(".pdf"):
                pages = render_pdf_to_images(asset["stored_path"], str(pages_dir), cfg.ingest.pdf_dpi)
            else:
                meta = normalize_image_to_page(asset["stored_path"], str(pages_dir))
                pages = [meta]
            _insert_page_records(asset_id, pages)
            _set_stage(asset_id, "rendered")
            current_stage = "rendered"
        else:
            pages_rows = execute(asset_service.get_db_path(), "SELECT page_num, image_path, width, height FROM asset_pages WHERE asset_id = ? ORDER BY page_num;", (asset_id,), fetchall=True) or []
            pages = pages_rows

        if _should_run(current_stage, "ocr_done"):
            ocr_engine, ocr_warning, ocr_engine_name = get_ocr_engine(lang=cfg.ingest.ocr_lang, config=cfg)
            for page in pages:
                page_num = page["page_num"] if isinstance(page, dict) else page.get("page_num")
                image_path = page["image_path"] if isinstance(page, dict) else page.get("image_path")
                ocr_json = ocr_engine.ocr_page(image_path, page_num)
                blocks = ocr_json.get("blocks", [])
                chars = sum(len(b.get("text", "")) for b in blocks)
                logger.info(
                    "OCR page processed",
                    extra={
                        "asset_id": asset_id,
                        "page_num": page_num,
                        "ocr_engine": ocr_engine_name,
                        "blocks": len(blocks),
                        "chars": chars,
                    },
                )
                ocr_path = save_ocr_json(ocr_json, asset_id, ocr_dir, page_num)
                stats = analyze_ocr_stats(ocr_json)
                if not ocr_json.get("blocks"):
                    stats["needs_caption"] = 1
                _insert_ocr_record(asset_id, page_num, ocr_path, stats)
            engine_msg = ocr_warning or f"OCR engine used: {ocr_engine_name}"
            _set_stage(asset_id, "ocr_done", error=engine_msg, ocr_engine=ocr_engine_name, warning=ocr_warning)
            current_stage = "ocr_done"

        chunks_all: List[dict] = []
        if _should_run(current_stage, "chunked"):
            ocr_records = execute(asset_service.get_db_path(), "SELECT page_num, ocr_json_path FROM asset_ocr_pages WHERE asset_id = ? ORDER BY page_num;", (asset_id,), fetchall=True) or []
            for rec in ocr_records:
                page_num = rec["page_num"] if isinstance(rec, dict) else rec.get("page_num")
                ocr_json = json.loads(Path(rec["ocr_json_path"]).read_text(encoding="utf-8"))
                chunks = chunk_ocr_blocks(
                    subject_id,
                    asset_id,
                    page_num,
                    ocr_json,
                    cfg.ingest.max_chunk_chars,
                    cfg.ingest.min_chunk_chars,
                    cfg.ingest.overlap_blocks,
                )
                chunks_all.extend(chunks)
            write_chunks_jsonl(chunks_all, chunks_dir / f"{asset_id}.jsonl")
            _upsert_chunks(chunks_all)
            _set_stage(asset_id, "chunked")
            current_stage = "chunked"
        else:
            chunks_path = chunks_dir / f"{asset_id}.jsonl"
            if chunks_path.exists():
                for line in chunks_path.read_text(encoding="utf-8").splitlines():
                    try:
                        chunks_all.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if _should_run(current_stage, "embedded"):
            embedder = Embedder(config=cfg)
            vectors = embedder.embed_texts([c["text"] for c in chunks_all])  # type: ignore[arg-type]
            payloads = []
            ids = []
            page_lookup = {p["page_num"]: p for p in pages}
            for chunk, vec in zip(chunks_all, vectors):
                page_meta = page_lookup.get(chunk["page_num"], {})
                identity = f"{chunk['subject_id']}:{chunk['asset_id']}:{chunk['page_num']}:{chunk.get('chunk_id', chunk.get('start_block'))}"
                point_id = make_point_uuid(identity)
                payloads.append(
                    {
                        "source_type": "slide",
                        "subject_id": subject_id,
                        "asset_id": asset_id,
                        "page_num": chunk["page_num"],
                        "image_path": page_meta.get("image_path"),
                        "source": asset.get("original_filename"),
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "preview": chunk["text"][:240],
                    }
                )
                ids.append(point_id)
            store = QdrantStore()
            store.upsert_chunks(vectors, payloads, ids)
            _set_stage(asset_id, "embedded")
            current_stage = "embedded"

        if _should_run(current_stage, "indexed"):
            _set_stage(asset_id, "indexed")

    except Exception as exc:  # pragma: no cover - pipeline error path
        _set_stage(asset_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def process_subject_new_assets(subject_id: str, config=None, *, force: bool = False, limit: int | None = None) -> dict:
    cfg = config or load_config()
    assets = asset_service.list_assets(subject_id)
    summary = {"indexed": 0, "skipped_missing": 0, "failed": 0, "details": [], "processed": 0}
    for asset in assets:
        status = asset_service.get_index_status(asset["asset_id"])
        stage = status.get("stage") if status else asset.get("status")
        if stage == "indexed" and not force:
            continue
        if limit is not None and summary["processed"] >= limit:
            break
        summary["processed"] += 1
        try:
            process_asset(subject_id, asset, cfg, force=force)
            latest = asset_service.get_index_status(asset["asset_id"])
            latest_stage = latest.get("stage") if latest else None
            if latest_stage == "missing":
                summary["skipped_missing"] += 1
            elif latest_stage == "indexed":
                summary["indexed"] += 1
            else:
                summary["failed"] += 1
            summary["details"].append({"asset_id": asset["asset_id"], "stage": latest_stage, "error": latest.get("error") if latest else None})
        except Exception as exc:  # continue on failures
            asset_service.upsert_index_status(asset["asset_id"], "failed", error=str(exc))
            summary["failed"] += 1
            summary["details"].append({"asset_id": asset["asset_id"], "stage": "failed", "error": str(exc)})
            continue
    return summary


__all__ = ["process_asset", "process_subject_new_assets"]
