from types import SimpleNamespace
from pathlib import Path

from rag_assistant.config import Settings, AppConfig, DatabaseConfig, QdrantConfig, LoggingConfig, IngestConfig, RetrievalConfig, LLMConfig, EmbeddingsConfig
from rag_assistant.retrieval.vector_store import qdrant as qstore


def _dummy_config(db_path: Path):
    return Settings(
        app=AppConfig(),
        database=DatabaseConfig(sqlite_path=str(db_path)),
        qdrant=QdrantConfig(collection="test_retry", vector_size=384, url="http://localhost:6333"),
        logging=LoggingConfig(),
        ingest=IngestConfig(),
        retrieval=RetrievalConfig(),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(),
    )


def test_retry_without_subject_returns_hits(tmp_path, monkeypatch):
    called = {"queries": []}

    class Client:
        def __init__(self):
            self.phase = 0

        def search(self, **kwargs):
            called["queries"].append(kwargs.get("query_vector"))
            if kwargs.get("query_filter"):
                return []
            return [SimpleNamespace(id="p1", score=0.9, payload={"chunk_id": "c1", "text": "hello", "page_num": 1})]

        def get_collections(self):
            return SimpleNamespace(collections=[])

        def create_collection(self, **kwargs):
            return None

        def get_collection(self, *args, **kwargs):
            return SimpleNamespace(vectors=SimpleNamespace(size=384), config=None, points_count=0)

    monkeypatch.setattr(qstore, "QdrantClient", lambda url=None, api_key=None: Client())
    monkeypatch.setattr(qstore, "load_config", lambda: _dummy_config(tmp_path / "db.sqlite"))
    store = qstore.QdrantStore()
    hits = store.search([0.1, 0.2], subject_id="subj", limit=5)
    assert hits
    assert hits[0]["chunk_id"] == "c1"
