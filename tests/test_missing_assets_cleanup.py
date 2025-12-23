from pathlib import Path

import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import init_db
from rag_assistant.ingest import pipeline
from rag_assistant.services import asset_service, cleanup_service, subject_service


def setup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    cfg = load_config()
    init_db(db_path)
    return cfg


def test_missing_assets_marked_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = setup_env(tmp_path, monkeypatch)
    subject = subject_service.create_subject("Cleanup Test")

    # Create asset pointing to missing file
    missing_path = tmp_path / "does_not_exist.pdf"
    asset = asset_service.add_asset(subject["subject_id"], "missing.pdf", b"dummy", "application/pdf")
    # Overwrite stored_path to nonexistent
    asset_service.execute(asset_service.get_db_path(), "UPDATE assets SET stored_path=? WHERE asset_id=?;", (str(missing_path), asset["asset_id"]))

    result = pipeline.process_subject_new_assets(subject["subject_id"], cfg)
    assert result["skipped_missing"] >= 1
    status = asset_service.get_index_status(asset["asset_id"])
    assert status["stage"] == "missing"

    missing = cleanup_service.list_missing_assets(subject["subject_id"])
    assert any(m["asset_id"] == asset["asset_id"] for m in missing)

    cleanup_service.remove_assets(subject["subject_id"], [asset["asset_id"]], remove_vectors=False)
    remaining = asset_service.list_assets(subject["subject_id"])
    assert all(a["asset_id"] != asset["asset_id"] for a in remaining)
