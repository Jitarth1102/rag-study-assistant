"""Health checks for local dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

from rag_assistant.db.base import get_connection
from rag_assistant.ingest.ocr.selftest import run_ocr_selftest
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore


@dataclass
class HealthResult:
    ok: bool
    detail: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, **self.detail}


def check_db(cfg) -> dict:
    path = Path(cfg.database.sqlite_path)
    try:
        with get_connection(path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        required = {"subjects", "assets", "chunks", "asset_pages", "asset_ocr_pages", "asset_index_status"}
        missing = sorted(required - tables)
        return HealthResult(ok=len(missing) == 0, detail={"path": str(path), "tables": sorted(tables), "missing": missing}).to_dict()
    except Exception as exc:
        return HealthResult(ok=False, detail={"path": str(path), "error": str(exc)}).to_dict()


def check_qdrant(cfg, store_factory: Optional[Callable[[], QdrantStore]] = None) -> dict:
    factory = store_factory or QdrantStore
    try:
        store = factory()
    except Exception as exc:
        return HealthResult(ok=False, detail={"error": f"connect/ensure failed: {exc}"}).to_dict()

    detail: dict = {"url": getattr(cfg.qdrant, "url", None), "collection": getattr(store, "collection", None)}
    try:
        store.client.get_collections()
    except Exception as exc:
        detail["error"] = f"collections fetch failed: {exc}"
        return HealthResult(ok=False, detail=detail).to_dict()

    size_ok = True
    try:
        info = store.client.get_collection(store.collection)
        current_size = store._get_collection_vector_size(info)
        expected_size = int(getattr(store, "vector_size", 0))
        detail["collection_vector_size"] = current_size
        detail["expected_vector_size"] = expected_size
        if current_size is not None and expected_size and current_size != expected_size:
            size_ok = False
            detail["error"] = f"vector size mismatch: {current_size} != {expected_size}"
    except Exception as exc:
        detail["error"] = f"collection fetch failed: {exc}"
        return HealthResult(ok=False, detail=detail).to_dict()

    try:
        detail["points"] = store.get_collection_point_count()
    except Exception as exc:
        detail["points_error"] = str(exc)

    return HealthResult(ok=size_ok, detail=detail).to_dict()


def check_ollama(cfg, session=None) -> dict:
    provider = getattr(cfg.llm, "provider", "").lower()
    if provider != "ollama":
        return HealthResult(ok=True, detail={"skipped": True, "provider": provider}).to_dict()
    client = session or requests
    url = f"{cfg.llm.base_url.rstrip('/')}/api/tags"
    try:
        resp = client.get(url, timeout=5)
        if resp.status_code != 200:
            return HealthResult(ok=False, detail={"url": url, "status": resp.status_code, "text": getattr(resp, "text", "")}).to_dict()
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {}
        return HealthResult(ok=True, detail={"url": url, "models": data.get("models") or data.get("tags") or data}).to_dict()
    except Exception as exc:
        return HealthResult(ok=False, detail={"url": url, "error": str(exc)}).to_dict()


def run_ocr_check(cfg, *, runner: Callable = None) -> dict:
    run = runner or run_ocr_selftest
    try:
        res = run(cfg)
        return HealthResult(ok=True, detail={"result": res}).to_dict()
    except Exception as exc:
        return HealthResult(ok=False, detail={"error": str(exc)}).to_dict()


def run_all_checks(cfg, *, include_ocr: bool = False, store_factory=None, ollama_session=None) -> dict:
    results = {
        "db": check_db(cfg),
        "qdrant": check_qdrant(cfg, store_factory=store_factory),
        "ollama": check_ollama(cfg, session=ollama_session),
    }
    if include_ocr:
        results["ocr"] = run_ocr_check(cfg)
    return results


__all__ = ["check_db", "check_qdrant", "check_ollama", "run_ocr_check", "run_all_checks"]
