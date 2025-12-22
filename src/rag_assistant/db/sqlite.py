"""SQLite helpers."""

from __future__ import annotations

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
    """Ensure required columns exist on chunks table; add if missing."""
    required_cols = ["asset_id", "subject_id", "page_num", "start_block", "end_block", "bbox_json"]
    for col in required_cols:
        if not has_column(db_path, "chunks", col):
            with get_connection(db_path) as conn:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT;")
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
    # Ensure chunks columns even if migration file missing
    ensure_chunks_columns(db_path)


def delete_asset_dependent_rows(db_path: Path, asset_id: str) -> None:
    with get_connection(db_path) as conn:
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
