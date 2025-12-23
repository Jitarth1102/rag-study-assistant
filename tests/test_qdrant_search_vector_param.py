from types import SimpleNamespace

from rag_assistant.retrieval.vector_store.qdrant import search_points


def test_search_points_uses_query_vector(monkeypatch):
    called = {}

    class Client:
        def search(self, **kwargs):
            called.update(kwargs)
            return [SimpleNamespace(id="1", score=0.9, payload={"chunk_id": "c1", "text": "hi", "page_num": 1})]

    res = search_points(Client(), "col", [0.1, 0.2], 5, query_filter=None, with_payload=True)
    assert res
    assert called["query_vector"] == [0.1, 0.2]
    assert called["with_payload"] is True


def test_query_points_uses_query_vector(monkeypatch):
    called = {}

    class Client:
        def query_points(self, **kwargs):
            called.update(kwargs)
            return [SimpleNamespace(id="1", score=0.8, payload={"chunk_id": "c2", "text": "hello", "page_num": 2})]

    res = search_points(Client(), "col", [0.3, 0.4], 3, query_filter={"foo": "bar"}, with_payload=True)
    assert res
    assert called.get("query_vector") == [0.3, 0.4] or called.get("query") == [0.3, 0.4]
    assert called.get("with_payload") is True
    assert called.get("filter") == {"foo": "bar"} or called.get("query_filter") == {"foo": "bar"}
