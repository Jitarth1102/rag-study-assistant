from types import SimpleNamespace

from rag_assistant.rag import answerer
from rag_assistant.rag.judge import JudgeDecision
from rag_assistant.web.search_client import WebResult


def test_answerer_web_fallback(monkeypatch):
    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.0] * 384]

    class DummyStore:
        collection = "col"

        def __init__(self):
            pass

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return []

        def search_notes(self, vector, subject_id, limit):
            return []

    cfg = SimpleNamespace(
        web=SimpleNamespace(
            enabled=True,
            provider="serpapi",
            api_key="key",
            max_results=5,
            timeout_s=10,
            min_rag_score_to_skip_web=0.0,
            min_rag_hits_to_skip_web=0,
            max_web_queries_per_question=2,
        ),
        embeddings=SimpleNamespace(vector_size=384),
        retrieval=SimpleNamespace(top_k=3, neighbor_window=1, max_neighbor_chunks=12, min_score=0.0),
        database=SimpleNamespace(sqlite_path="/tmp/db.sqlite"),
        app=SimpleNamespace(data_root="/tmp"),
    )

    monkeypatch.setattr(answerer, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda *a, **k: [])
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "answer with web")
    monkeypatch.setattr(answerer.judge, "should_search_web", lambda *a, **k: JudgeDecision(do_search=True, reason="no_hits", suggested_queries=["q"]))
    monkeypatch.setattr(
        answerer.search_client,
        "search",
        lambda query, config=None, **kwargs: [WebResult(title="Web", url="http://example.com", snippet="Snippet", source="example.com")],
    )

    res = answerer.ask("subj", "question", 3, config=cfg)
    assert res["used_web"] is True
    web_cites = [c for c in res["citations"] if c.get("type") == "web"]
    assert web_cites
