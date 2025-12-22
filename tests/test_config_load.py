from pathlib import Path

from rag_assistant.config import load_config


def test_load_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
app:
  name: test-app
  environment: test
  data_root: ./data_test
  logs_dir: ./logs_test

database:
  sqlite_path: ./data_test/db/test.db

qdrant:
  host: localhost
  port: 6333
  collection_name: test_collection

logging:
  level: DEBUG
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", "./data_env")
    settings = load_config(config_path)
    assert settings.app.name == "test-app"
    assert Path(settings.app.data_root).exists()
    assert Path(settings.app.logs_dir).exists()
    assert Path(settings.database.sqlite_path).parent.exists()
    assert settings.qdrant.collection_name == "test_collection"
