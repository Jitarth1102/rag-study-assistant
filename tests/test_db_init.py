from pathlib import Path
import sqlite3

from rag_assistant.db.sqlite import init_db


def test_init_db_creates_tables(tmp_path: Path):
    db_path = tmp_path / "db" / "test.db"
    init_db(db_path)
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    expected = {
        "documents",
        "chunks",
        "runs",
        "quiz_attempts",
        "subjects",
        "assets",
        "asset_index_status",
        "asset_pages",
        "asset_ocr_pages",
        "schema_migrations",
    }
    assert expected.issubset(tables)
