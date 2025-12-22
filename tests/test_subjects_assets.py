from hashlib import sha256
from pathlib import Path

import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import init_db
from rag_assistant.services import asset_service, subject_service


def setup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    load_config()  # create directories
    init_db(db_path)
    return db_path


def test_create_subject_inserts_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    setup_env(tmp_path, monkeypatch)
    subject = subject_service.create_subject("Machine Learning Fall 2025")
    subjects = subject_service.list_subjects()
    assert any(s["subject_id"] == subject["subject_id"] for s in subjects)


def test_add_asset_saves_file_and_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    setup_env(tmp_path, monkeypatch)
    subject = subject_service.create_subject("Physics")
    file_bytes = b"example content"
    asset = asset_service.add_asset(subject["subject_id"], "notes.txt", file_bytes, "text/plain")
    expected_id = sha256(file_bytes).hexdigest()[:16]
    assert asset["asset_id"] == expected_id
    assert Path(asset["stored_path"]).exists()

    assets = asset_service.list_assets(subject["subject_id"])
    assert len(assets) == 1

    duplicate = asset_service.add_asset(subject["subject_id"], "notes.txt", file_bytes, "text/plain")
    assets_after = asset_service.list_assets(subject["subject_id"])
    assert duplicate["asset_id"] == asset["asset_id"]
    assert len(assets_after) == 1
