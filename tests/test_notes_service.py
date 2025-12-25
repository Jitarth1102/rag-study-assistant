from pathlib import Path

import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, init_db
from rag_assistant.services import asset_service, notes_service, subject_service
from rag_assistant.web.search_client import WebResult
from rag_assistant.rag.judge import JudgeDecision


def _setup_subject_and_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    cfg = load_config()
    init_db(db_path)
    subject = subject_service.create_subject("Test Subject")
    asset = asset_service.add_asset(subject["subject_id"], "sample.pdf", b"file-bytes", "application/pdf")
    execute(
        db_path,
        "INSERT INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, '{}', 0, 0, 0.0);",
        ("chunk1", subject["subject_id"], asset["asset_id"], 1, "Intro to testing"),
    )
    return cfg, subject, asset, db_path


def test_generate_notes_creates_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.1] * 2 for _ in texts]

    store_calls = {"upserts": 0, "payloads": [], "deleted": []}

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            store_calls["deleted"].append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            store_calls["upserts"] += 1
            store_calls["payloads"] = payloads

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg: "## Heading\nSome bullet point")

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    notes_row = execute(db_path, "SELECT * FROM notes WHERE notes_id = ?;", (res["notes_id"],), fetchone=True)
    chunks = execute(db_path, "SELECT * FROM notes_chunks WHERE notes_id = ?;", (res["notes_id"],), fetchall=True)

    assert notes_row is not None
    assert notes_row["version"] == 1
    assert chunks
    assert store_calls["upserts"] == 1
    assert store_calls["payloads"][0]["source_type"] == "notes"
    assert store_calls["payloads"][0]["source_label"] == "Generated Notes"
    assert store_calls["payloads"][0]["version"] == 1
    assert store_calls["deleted"]


def test_update_notes_increments_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.2] * 2 for _ in texts]

    class DummyStore:
        def __init__(self):
            self.deleted = []
            self.upserts = 0
            self.payloads = []

        def delete_by_notes_id(self, notes_id):
            self.deleted.append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            self.upserts += 1
            self.payloads = payloads

    store = DummyStore()
    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: store)
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg: "## Intro\nDetails")

    initial = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    updated = notes_service.update_notes(initial["notes_id"], "# Updated\nNew content", config=cfg)

    row = execute(db_path, "SELECT version, markdown FROM notes WHERE notes_id = ?;", (initial["notes_id"],), fetchone=True)
    assert updated["version"] == 2
    assert row["version"] == 2
    assert "Updated" in row["markdown"]
    assert store.deleted  # deletion before re-upsert
    assert store.upserts >= 1
    assert store.payloads[0]["source_label"] == "From User Notes"
    assert store.payloads[0]["version"] == 2


def test_web_augmentation_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.web.max_web_queries_per_question = 1

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.3] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    counter = {"calls": 0}

    def fake_search(query, config=None, allowlist=None, blocklist=None):
        counter["calls"] += 1
        return [WebResult(title=query, url="http://example.com", snippet="snippet", source="example")]

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg: "## With web\ndata")
    monkeypatch.setattr(notes_service.search_client, "search", fake_search)
    monkeypatch.setattr(
        notes_service.judge,
        "should_search_web",
        lambda *a, **k: JudgeDecision(do_search=True, reason="force", suggested_queries=["q1", "q2"]),
    )

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    assert counter["calls"] == 1  # bounded by max_web_queries_per_question
    assert res["used_web"] in {True, False}
    notes_row = execute(db_path, "SELECT meta_json FROM notes WHERE notes_id = ?;", (res["notes_id"],), fetchone=True)
    assert notes_row is not None
