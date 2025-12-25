"""SQLite helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from rag_assistant.db.base import ensure_parent_dir, get_connection

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def init_db(db_path: Path) -> None:
    """Initialize the SQLite database using the schema file."""
    ensure_parent_dir(db_path)
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema_sql = f.read()
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    apply_migrations(db_path)


def execute(db_path: Path, sql: str, params: Iterable[Any] | dict[str, Any] = (), *, fetchone: bool = False, fetchall: bool = False):
    """Execute a SQL statement and optionally fetch results."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        if fetchone:
            row = cursor.fetchone()
            return dict(row) if row else None
        if fetchall:
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    return None


def has_column(db_path: Path, table: str, column: str) -> bool:
    with get_connection(db_path) as conn:
        info = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(row[1] == column for row in info)


def ensure_chunks_columns(db_path: Path) -> None:
    """Ensure required columns exist on chunks table; add/backfill if missing."""
    required_cols = ["chunk_id", "asset_id", "subject_id", "page_num", "start_block", "end_block", "bbox_json", "created_at"]
    added_chunk_id = False
    for col in required_cols:
        if not has_column(db_path, "chunks", col):
            with get_connection(db_path) as conn:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT;")
                conn.commit()
            if col == "chunk_id":
                added_chunk_id = True
    if added_chunk_id:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT rowid, asset_id, page_num, start_block, end_block FROM chunks WHERE chunk_id IS NULL OR chunk_id = '';"
            ).fetchall()
            for row in rows:
                asset_id = row["asset_id"] or ""
                page_num = row["page_num"] if row["page_num"] is not None else ""
                start_block = row["start_block"]
                end_block = row["end_block"]
                if start_block is not None and end_block is not None and start_block != "" and end_block != "":
                    identity = f"{asset_id}:{page_num}:{start_block}:{end_block}"
                else:
                    identity = f"legacy:{asset_id}:{page_num}:{row['rowid']}"
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                conn.execute("UPDATE chunks SET chunk_id = ? WHERE rowid = ?;", (chunk_id, row["rowid"]))
            conn.commit()
    # indexes
    with get_connection(db_path) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_subject_asset_page ON chunks(subject_id, asset_id, page_num);")
        conn.commit()


def ensure_asset_status_columns(db_path: Path) -> None:
    required_cols = ["ocr_engine", "warning"]
    for col in required_cols:
        if not has_column(db_path, "asset_index_status", col):
            with get_connection(db_path) as conn:
                conn.execute(f"ALTER TABLE asset_index_status ADD COLUMN {col} TEXT;")
                conn.commit()


def apply_migrations(db_path: Path) -> None:
    """Apply SQL migrations once and run column checks for legacy DBs."""
    with get_connection(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (id TEXT PRIMARY KEY, applied_at REAL NOT NULL);"
        )
        applied = {row[0] for row in conn.execute("SELECT id FROM schema_migrations;").fetchall()}
        conn.commit()

    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for mig in migrations:
        mig_id = mig.stem
        if mig_id in applied:
            continue
        if mig.name == "004_chunks_add_asset_id.sql":
            ensure_chunks_columns(db_path)
            with get_connection(db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO schema_migrations (id, applied_at) VALUES (?, strftime('%s','now'));", (mig_id,))
                conn.commit()
            continue
        sql = mig.read_text(encoding="utf-8")
        if ".read" in sql:
            # Skip shell-style directives; schema already applied
            with get_connection(db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO schema_migrations (id, applied_at) VALUES (?, strftime('%s','now'));", (mig_id,))
                conn.commit()
            continue
        with get_connection(db_path) as conn:
            conn.executescript(sql)
            conn.execute("INSERT OR REPLACE INTO schema_migrations (id, applied_at) VALUES (?, strftime('%s','now'));", (mig_id,))
            conn.commit()
    # Ensure columns even if migration file missing
    ensure_chunks_columns(db_path)
    ensure_asset_status_columns(db_path)


def delete_asset_dependent_rows(db_path: Path, asset_id: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM notes_chunks WHERE asset_id = ?;", (asset_id,))
        conn.execute("DELETE FROM notes WHERE asset_id = ?;", (asset_id,))
        conn.execute("DELETE FROM asset_pages WHERE asset_id = ?;", (asset_id,))
        conn.execute("DELETE FROM asset_ocr_pages WHERE asset_id = ?;", (asset_id,))
        if has_column(db_path, "chunks", "asset_id"):
            conn.execute("DELETE FROM chunks WHERE asset_id = ?;", (asset_id,))
        conn.execute("DELETE FROM asset_index_status WHERE asset_id = ?;", (asset_id,))
        conn.commit()


def delete_asset(db_path: Path, asset_id: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM assets WHERE asset_id = ?;", (asset_id,))
        conn.commit()


def list_assets_with_missing_files(db_path: Path, subject_id: str) -> list[dict]:
    rows = execute(
        db_path,
        "SELECT asset_id, subject_id, original_filename, stored_path, status FROM assets WHERE subject_id = ?;",
        (subject_id,),
        fetchall=True,
    )
    missing = []
    for row in rows or []:
        path = Path(row["stored_path"])
        if not path.exists():
            missing.append(row)
    return missing


__all__ = ["init_db", "SCHEMA_PATH", "execute"]
