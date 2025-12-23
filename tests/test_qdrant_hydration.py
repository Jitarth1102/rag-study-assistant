from types import SimpleNamespace
from pathlib import Path

from rag_assistant.config import Settings, AppConfig, DatabaseConfig, QdrantConfig, LoggingConfig, IngestConfig, RetrievalConfig, LLMConfig, EmbeddingsConfig
from rag_assistant.db.base import get_connection
from rag_assistant.retrieval.vector_store import qdrant as qstore


def _dummy_config(db_path: Path):
    return Settings(
        app=AppConfig(),
        database=DatabaseConfig(sqlite_path=str(db_path)),
        qdrant=QdrantConfig(collection="test_col", vector_size=384, url="http://localhost:6333"),
        logging=LoggingConfig(),
        ingest=IngestConfig(),
        retrieval=RetrievalConfig(),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(),
    )


class DummyClient:
    def __init__(self, points):
        self.points = points

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def create_collection(self, **kwargs):
        return None

    def get_collection(self, *args, **kwargs):
        return SimpleNamespace(vectors=SimpleNamespace(size=384), config=None, points_count=0)

    def search(self, **kwargs):
        return self.points


def test_qdrant_hydrates_from_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        conn.execute(
            "INSERT INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, '{}', 0, 0, 0.0);",
            ("cid1", "subj1", "asset1", 1, "hello world"),
        )
        conn.commit()

    points = [SimpleNamespace(id="p1", score=0.9, payload={"chunk_id": "cid1", "asset_id": "asset1", "subject_id": "subj1"})]
    monkeypatch.setattr(qstore, "QdrantClient", lambda url=None, api_key=None: DummyClient(points))
    monkeypatch.setattr(qstore, "load_config", lambda: _dummy_config(db_path))

    store = qstore.QdrantStore()
    hits = store.search([0.1] * 384, subject_id="subj1", limit=5)
    assert hits
    hit = hits[0]
    assert hit.get("text") == "hello world"
    assert hit.get("page_num") == 1
    assert hit.get("chunk_id") == "cid1"


def test_qdrant_drops_hits_without_chunk_id(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        conn.commit()

    points = [SimpleNamespace(id="p1", score=0.9, payload={"text": "orphan"})]
    monkeypatch.setattr(qstore, "QdrantClient", lambda url=None, api_key=None: DummyClient(points))
    monkeypatch.setattr(qstore, "load_config", lambda: _dummy_config(db_path))

    store = qstore.QdrantStore()
    hits = store.search([0.1] * 384, subject_id=None, limit=5)
    assert hits == []
