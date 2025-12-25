from types import SimpleNamespace

from rag_assistant.rag import answerer
from rag_assistant.rag.judge import JudgeDecision
from rag_assistant.web.search_client import WebResult


def _cfg_with_web(max_queries=1):
    return SimpleNamespace(
        web=SimpleNamespace(
            enabled=True,
            provider="serpapi",
            api_key="key",
            max_results=5,
            timeout_s=5,
            min_rag_score_to_skip_web=0.0,
            min_rag_hits_to_skip_web=0,
            max_web_queries_per_question=max_queries,
            force_even_if_rag_strong=False,
            allowed_domains=[],
            blocked_domains=[],
        ),
        embeddings=SimpleNamespace(vector_size=384),
        retrieval=SimpleNamespace(top_k=3, neighbor_window=1, max_neighbor_chunks=12, min_score=0.0),
        database=SimpleNamespace(sqlite_path="/tmp/db.sqlite"),
        app=SimpleNamespace(data_root="/tmp"),
    )


def test_max_web_queries_enforced(monkeypatch):
    counter = {"calls": 0}

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.0] * 384]

    class DummyStore:
        collection = "col"

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return []

        def search_notes(self, vector, subject_id, limit):
            return []

    monkeypatch.setattr(answerer, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda *a, **k: [])
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "answer with web")
    monkeypatch.setattr(answerer.judge, "should_search_web", lambda *a, **k: JudgeDecision(do_search=True, reason="no_hits", suggested_queries=["q1", "q2"]))

    def fake_search(query, config=None, allowlist=None, blocklist=None):
        counter["calls"] += 1
        return [WebResult(title=query, url="http://example.com", snippet="s", source="example.com")]

    monkeypatch.setattr(answerer.search_client, "search", fake_search)

    res = answerer.ask("s", "q", 3, config=_cfg_with_web(max_queries=1))
    assert res["used_web"] is True
    dbg = res.get("debug") or {}
    assert dbg.get("web_queries_used") == 1
    assert counter["calls"] == 1


def test_web_failure_sets_error(monkeypatch):
    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.0] * 384]

    class DummyStore:
        collection = "col"

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return []

        def search_notes(self, vector, subject_id, limit):
            return []

    monkeypatch.setattr(answerer, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda *a, **k: [])
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "answer fallback")
    monkeypatch.setattr(answerer.judge, "should_search_web", lambda *a, **k: JudgeDecision(do_search=True, reason="no_hits", suggested_queries=["q1"]))
    monkeypatch.setattr(answerer.search_client, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))

    res = answerer.ask("s", "q", 3, config=_cfg_with_web(max_queries=2))
    assert res["used_web"] is False
    dbg = res.get("debug") or {}
    assert "fail" in (dbg.get("web_error") or "")
