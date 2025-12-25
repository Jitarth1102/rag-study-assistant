from pathlib import Path
import sqlite3

from rag_assistant.db.sqlite import init_db


def test_ingest_tables_exist(tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    expected = {"asset_index_status", "asset_pages", "asset_ocr_pages", "chunks", "notes", "notes_chunks"}
    assert expected.issubset(tables)
