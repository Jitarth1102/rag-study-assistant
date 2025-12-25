from pathlib import Path

import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, init_db
from rag_assistant.services import asset_service, cleanup_service, subject_service


def _seed_asset_with_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    cfg = load_config()
    init_db(db_path)
    subj = subject_service.create_subject("Test Subject")
    asset = asset_service.add_asset(subj["subject_id"], "orig.pdf", b"bytes", "application/pdf")
    # seed rows
    execute(
        db_path,
        "INSERT INTO asset_pages (asset_id, page_num, image_path, created_at) VALUES (?, ?, ?, 0.0);",
        (asset["asset_id"], 1, "/tmp/img.png"),
    )
    execute(
        db_path,
        "INSERT INTO asset_ocr_pages (asset_id, page_num, ocr_json_path, text_len, avg_conf, needs_caption, created_at) VALUES (?, ?, ?, ?, ?, 0, 0.0);",
        (asset["asset_id"], 1, "/tmp/ocr.json", 10, 0.9),
    )
    execute(
        db_path,
        "INSERT INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, '{}', 0, 0, 0.0);",
        ("chunk1", subj["subject_id"], asset["asset_id"], 1, "text"),
    )
    execute(
        db_path,
        "INSERT INTO notes (notes_id, subject_id, asset_id, version, markdown, generated_by, created_at) VALUES (?, ?, ?, 1, 'md', 'user', 0.0);",
        ("notes1", subj["subject_id"], asset["asset_id"]),
    )
    execute(
        db_path,
        "INSERT INTO notes_chunks (notes_chunk_id, notes_id, subject_id, asset_id, section_title, text, created_at) VALUES (?, ?, ?, ?, 'sec', 'note text', 0.0);",
        ("nchunk1", "notes1", subj["subject_id"], asset["asset_id"]),
    )
    return cfg, subj, asset, db_path


def test_delete_asset_removes_db_and_calls_qdrant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subj, asset, db_path = _seed_asset_with_notes(tmp_path, monkeypatch)

    calls = {"asset": [], "notes": []}

    class DummyStore:
        def delete_by_asset_id(self, asset_id):
            calls["asset"].append(asset_id)

        def delete_by_notes_id(self, notes_id):
            calls["notes"].append(notes_id)

    monkeypatch.setattr(cleanup_service, "QdrantStore", lambda: DummyStore())

    cleanup_service.remove_assets(subj["subject_id"], [asset["asset_id"]], remove_vectors=True)

    for table in ["asset_pages", "asset_ocr_pages", "chunks", "asset_index_status", "notes", "notes_chunks"]:
        rows = execute(db_path, f"SELECT * FROM {table};", fetchall=True)
        assert rows == [] or rows is None
    assert calls["asset"] == [asset["asset_id"]]
    assert calls["notes"] == ["notes1"]


def test_replace_asset_deletes_then_adds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subj, asset, db_path = _seed_asset_with_notes(tmp_path, monkeypatch)

    class DummyStore:
        def delete_by_asset_id(self, asset_id):
            pass

        def delete_by_notes_id(self, notes_id):
            pass

    monkeypatch.setattr(cleanup_service, "QdrantStore", lambda: DummyStore())
    # replace: delete then add new asset
    cleanup_service.remove_assets(subj["subject_id"], [asset["asset_id"]], remove_vectors=True)
    new_asset = asset_service.add_asset(subj["subject_id"], "new.pdf", b"newbytes", "application/pdf")
    # old asset gone
    assert execute(db_path, "SELECT * FROM assets WHERE asset_id = ?;", (asset["asset_id"],), fetchone=True) is None
    # new asset present
    assert execute(db_path, "SELECT * FROM assets WHERE asset_id = ?;", (new_asset["asset_id"],), fetchone=True)
