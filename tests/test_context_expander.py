from pathlib import Path

from rag_assistant.db.base import get_connection
from rag_assistant.retrieval.context_expander import expand_with_neighbors


def _seed_chunks(db_path: Path):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT,
                subject_id TEXT,
                asset_id TEXT,
                page_num INTEGER,
                text TEXT,
                bbox_json TEXT,
                start_block INTEGER,
                end_block INTEGER,
                created_at REAL
            );
            """
        )
        for page in range(1, 6):
            conn.execute(
                "INSERT INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, '{}', ?, ?, 0.0);",
                (f"c{page}", "subj", "asset1", page, f"content page {page}", 0, 0),
            )
        conn.commit()


def test_expand_with_neighbor_pages(tmp_path):
    db_path = tmp_path / "chunks.db"
    _seed_chunks(db_path)

    top_hit = {"chunk_id": "c3", "asset_id": "asset1", "page_num": 3, "text": "content page 3"}
    expanded = expand_with_neighbors([top_hit], window=1, max_extra=4, db_path=db_path)
    ids = [c["chunk_id"] for c in expanded]
    assert ids[0] == "c3"
    assert set(ids[1:]) == {"c2", "c4"}


def test_expand_respects_max_extra(tmp_path):
    db_path = tmp_path / "chunks.db"
    _seed_chunks(db_path)
    top_hit = {"chunk_id": "c3", "asset_id": "asset1", "page_num": 3, "text": "content page 3"}
    expanded = expand_with_neighbors([top_hit], window=2, max_extra=1, db_path=db_path)
    assert len(expanded) == 2  # original + one neighbor
    assert expanded[0]["chunk_id"] == "c3"
