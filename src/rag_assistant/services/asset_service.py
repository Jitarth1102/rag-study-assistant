"""Asset service for storing uploaded files and tracking metadata."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, init_db
from rag_assistant.services.subject_service import ensure_subject_dirs, get_subject

_DB_PATH: Path | None = None


def _db_path() -> Path:
    global _DB_PATH
    cfg = load_config()
    db_path = Path(cfg.database.sqlite_path)
    if _DB_PATH != db_path or not db_path.exists():
        init_db(db_path)
        _DB_PATH = db_path
    return db_path


def get_db_path() -> Path:
    return _db_path()


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe or "upload"


def _resolve_collision_path(directory: Path, filename: str) -> Path:
    target = directory / filename
    counter = 1
    while target.exists():
        target = directory / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
        counter += 1
    return target


def _get_asset(asset_id: str) -> Optional[dict]:
    sql = "SELECT * FROM assets WHERE asset_id = ?;"
    return execute(_db_path(), sql, (asset_id,), fetchone=True)


def get_asset(asset_id: str) -> Optional[dict]:
    return _get_asset(asset_id)


def add_asset(subject_id: str, uploaded_filename: str, file_bytes: bytes, mime_type: str | None) -> dict:
    """Store a file for a subject and record metadata.

    If an asset with the same content (sha256) already exists, return it.
    """
    subject = get_subject(subject_id)
    if subject is None:
        raise ValueError(f"Subject '{subject_id}' does not exist")

    sha_full = hashlib.sha256(file_bytes).hexdigest()
    asset_id = sha_full[:16]
    existing = _get_asset(asset_id)
    if existing:
        existing_path = Path(existing["stored_path"])
        if not existing_path.exists():
            existing_path.parent.mkdir(parents=True, exist_ok=True)
            existing_path.write_bytes(file_bytes)
        return existing

    storage_dir = ensure_subject_dirs(subject_id)
    sanitized_name = _sanitize_filename(uploaded_filename)
    target_path = _resolve_collision_path(storage_dir, sanitized_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(file_bytes)

    created_at = time.time()
    sql = (
        "INSERT INTO assets (asset_id, subject_id, original_filename, stored_path, sha256, size_bytes, mime_type, created_at, status, meta_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    params = (
        asset_id,
        subject_id,
        uploaded_filename,
        str(target_path),
        sha_full,
        len(file_bytes),
        mime_type,
        created_at,
        "stored",
        None,
    )
    execute(_db_path(), sql, params)
    return {
        "asset_id": asset_id,
        "subject_id": subject_id,
        "original_filename": uploaded_filename,
        "stored_path": str(target_path),
        "sha256": sha_full,
        "size_bytes": len(file_bytes),
        "mime_type": mime_type,
        "created_at": created_at,
        "status": "stored",
        "meta_json": None,
    }


def list_assets(subject_id: str) -> List[dict]:
    sql = "SELECT * FROM assets WHERE subject_id = ? ORDER BY created_at DESC;"
    return execute(_db_path(), sql, (subject_id,), fetchall=True) or []


def upsert_index_status(asset_id: str, stage: str, error: str | None = None) -> None:
    sql = "INSERT INTO asset_index_status (asset_id, stage, updated_at, error) VALUES (?, ?, ?, ?) ON CONFLICT(asset_id) DO UPDATE SET stage=excluded.stage, updated_at=excluded.updated_at, error=excluded.error;"
    execute(_db_path(), sql, (asset_id, stage, time.time(), error))
    # mirror to assets.status for quick UI reads
    execute(_db_path(), "UPDATE assets SET status = ? WHERE asset_id = ?;", (stage, asset_id))


def get_index_status(asset_id: str) -> Optional[dict]:
    return execute(_db_path(), "SELECT * FROM asset_index_status WHERE asset_id = ?;", (asset_id,), fetchone=True)


__all__ = ["add_asset", "list_assets", "get_asset", "upsert_index_status", "get_index_status", "get_db_path"]
