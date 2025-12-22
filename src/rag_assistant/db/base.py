"""Database base helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag_assistant.domain.errors import DatabaseError


def get_connection(db_path: Path) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as exc:
        raise DatabaseError(str(exc)) from exc


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["get_connection", "ensure_parent_dir"]
