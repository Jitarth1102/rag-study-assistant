from pathlib import Path
import sqlite3

from rag_assistant.db.sqlite import init_db


def test_chunks_adds_asset_id_on_migration(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL);")
        conn.commit()
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(chunks);")]
    assert "asset_id" in cols
    assert "subject_id" in cols
    assert "page_num" in cols
    assert "start_block" in cols
    assert "end_block" in cols
    assert "bbox_json" in cols
