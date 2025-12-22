"""SQLite helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from rag_assistant.db.base import ensure_parent_dir, get_connection

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_db(db_path: Path) -> None:
    """Initialize the SQLite database using the schema file."""
    ensure_parent_dir(db_path)
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema_sql = f.read()
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


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


__all__ = ["init_db", "SCHEMA_PATH", "execute"]
