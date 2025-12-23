from pathlib import Path
import sqlite3

from rag_assistant.db.sqlite import init_db


def test_chunk_id_migration(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (asset_id TEXT, subject_id TEXT, page_num INTEGER, text TEXT);")
        conn.execute("INSERT INTO chunks (asset_id, subject_id, page_num, text) VALUES ('a1','s1',1,'hello');")
        conn.commit()
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(chunks);")]
        assert "chunk_id" in cols
        conn.execute("INSERT INTO chunks (chunk_id, asset_id, subject_id, page_num, text, start_block, end_block, bbox_json, created_at) VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP);",
                     ("cid", "a1", "s1", 2, "text", 0, 0, "[]"))
