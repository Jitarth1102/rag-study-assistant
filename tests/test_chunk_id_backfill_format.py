import hashlib
import sqlite3
from pathlib import Path

from rag_assistant.db.sqlite import init_db


def test_chunk_id_backfill_matches_chunker_format(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE chunks (asset_id TEXT, subject_id TEXT, page_num INTEGER, text TEXT, start_block INTEGER, end_block INTEGER);"
        )
        conn.execute(
            "INSERT INTO chunks (asset_id, subject_id, page_num, text, start_block, end_block) VALUES (?,?,?,?,?,?);",
            ("a1", "s1", 1, "hello", 0, 0),
        )
        conn.commit()

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT chunk_id, asset_id, page_num, start_block, end_block FROM chunks LIMIT 1;").fetchone()
        assert row is not None
        expected = hashlib.sha256(f"{row['asset_id']}:{row['page_num']}:{row['start_block']}:{row['end_block']}".encode("utf-8")).hexdigest()[:20]
        assert row["chunk_id"] == expected
