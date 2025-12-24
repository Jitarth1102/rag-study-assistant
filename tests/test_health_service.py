from types import SimpleNamespace
from pathlib import Path

from rag_assistant.db.sqlite import init_db
from rag_assistant.services import health_service as hs


def test_check_db_ok(tmp_path):
    db_path = tmp_path / "db.sqlite"
    cfg = SimpleNamespace(database=SimpleNamespace(sqlite_path=str(db_path)))
    init_db(db_path)
    res = hs.check_db(cfg)
    assert res["ok"] is True
    assert "subjects" in res["tables"]


def test_check_qdrant_with_dummy_store():
    class DummyClient:
        def get_collections(self):
            class C:
                collections = []

            return C()

        def get_collection(self, name):
            return SimpleNamespace(vectors=SimpleNamespace(size=384), config=None)

    class DummyStore:
        def __init__(self):
            self.collection = "col"
            self.vector_size = 384
            self.client = DummyClient()

        def get_collection_point_count(self):
            return 3

        @staticmethod
        def _get_collection_vector_size(info):
            return getattr(info, "vectors", SimpleNamespace(size=None)).size

    cfg = SimpleNamespace(qdrant=SimpleNamespace(url="http://localhost:6333"))
    res = hs.check_qdrant(cfg, store_factory=DummyStore)
    assert res["ok"] is True
    assert res["points"] == 3


def test_check_qdrant_factory_failure():
    cfg = SimpleNamespace(qdrant=SimpleNamespace(url="http://localhost:6333"))

    def boom():
        raise RuntimeError("nope")

    res = hs.check_qdrant(cfg, store_factory=boom)
    assert res["ok"] is False
    assert "nope" in res["error"]


def test_check_ollama_skipped_when_not_provider():
    cfg = SimpleNamespace(llm=SimpleNamespace(provider="openai", base_url="http://localhost:11434"))
    res = hs.check_ollama(cfg)
    assert res["ok"] is True
    assert res["skipped"] is True


def test_check_ollama_ok(monkeypatch):
    class DummyResp:
        status_code = 200

        def json(self):
            return {"models": ["a", "b"]}

    class DummySession:
        @staticmethod
        def get(url, timeout=5):
            return DummyResp()

    cfg = SimpleNamespace(llm=SimpleNamespace(provider="ollama", base_url="http://localhost:11434"))
    res = hs.check_ollama(cfg, session=DummySession)
    assert res["ok"] is True
    assert res["models"] == ["a", "b"]


def test_run_all_checks_with_injected_dependencies(tmp_path):
    db_path = tmp_path / "db.sqlite"
    cfg = SimpleNamespace(
        database=SimpleNamespace(sqlite_path=str(db_path)),
        qdrant=SimpleNamespace(url="http://localhost:6333"),
        llm=SimpleNamespace(provider="ollama", base_url="http://localhost:11434"),
    )
    init_db(db_path)

    class DummyClient:
        def get_collections(self):
            class C:
                collections = []

            return C()

        def get_collection(self, name):
            return SimpleNamespace(vectors=SimpleNamespace(size=384), config=None)

    class DummyStore:
        def __init__(self):
            self.collection = "col"
            self.vector_size = 384
            self.client = DummyClient()

        def get_collection_point_count(self):
            return 0

        @staticmethod
        def _get_collection_vector_size(info):
            return getattr(info, "vectors", SimpleNamespace(size=None)).size

    class DummySession:
        @staticmethod
        def get(url, timeout=5):
            class R:
                status_code = 200

                @staticmethod
                def json():
                    return {"models": []}

            return R()

    res = hs.run_all_checks(cfg, include_ocr=False, store_factory=DummyStore, ollama_session=DummySession)
    assert res["db"]["ok"] is True
    assert res["qdrant"]["ok"] is True
    assert res["ollama"]["ok"] is True
