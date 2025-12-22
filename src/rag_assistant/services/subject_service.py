"""Subject service for managing study subjects."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, init_db

_DB_PATH: Path | None = None


def _db_path() -> Path:
    global _DB_PATH
    cfg = load_config()
    db_path = Path(cfg.database.sqlite_path)
    if _DB_PATH != db_path or not db_path.exists():
        init_db(db_path)
        _DB_PATH = db_path
    return db_path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", name.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "subject"


def _generate_subject_id(name: str, existing_ids: set[str]) -> str:
    base = _slugify(name)
    if base not in existing_ids:
        return base
    short_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()[:4]
    candidate = f"{base}-{short_hash}"
    counter = 1
    while candidate in existing_ids:
        candidate = f"{base}-{short_hash}-{counter}"
        counter += 1
    return candidate


def list_subjects() -> List[dict]:
    sql = "SELECT subject_id, name, created_at, meta_json FROM subjects ORDER BY created_at DESC;"
    rows = execute(_db_path(), sql, fetchall=True) or []
    return rows


def get_subject(subject_id: str) -> Optional[dict]:
    sql = "SELECT subject_id, name, created_at, meta_json FROM subjects WHERE subject_id = ?;"
    return execute(_db_path(), sql, (subject_id,), fetchone=True)


def ensure_subject_dirs(subject_id: str) -> Path:
    cfg = load_config()
    subject_dir = Path(cfg.app.data_root) / "subjects" / subject_id / "raw"
    subject_dir.mkdir(parents=True, exist_ok=True)
    return subject_dir


def create_subject(name: str) -> dict:
    if not name or not name.strip():
        raise ValueError("Subject name is required")

    existing = list_subjects()
    existing_ids = {row["subject_id"] for row in existing}
    subject_id = _generate_subject_id(name, existing_ids)
    created_at = time.time()
    sql = "INSERT INTO subjects (subject_id, name, created_at, meta_json) VALUES (?, ?, ?, ?);"
    execute(_db_path(), sql, (subject_id, name.strip(), created_at, None))
    ensure_subject_dirs(subject_id)
    return {"subject_id": subject_id, "name": name.strip(), "created_at": created_at, "meta_json": None}
