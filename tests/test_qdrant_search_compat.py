from types import SimpleNamespace

from rag_assistant.retrieval.vector_store.qdrant import search_points


class DummyClientSearch:
    def __init__(self, res):
        self._res = res

    def search(self, **kwargs):
        return self._res


class DummyClientQueryPoints:
    def __init__(self, res):
        self._res = res

    def query_points(self, **kwargs):
        return self._res


def test_search_path_uses_search():
    point = SimpleNamespace(id="1", score=0.9, payload={"a": 1})
    client = DummyClientSearch([point])
    res = search_points(client, "col", [0.1], 5)
    assert len(res) == 1
    assert res[0].id == "1"
    assert res[0].payload["a"] == 1


def test_search_path_uses_query_points():
    point = {"id": "2", "score": 0.8, "payload": {"b": 2}}
    client = DummyClientQueryPoints({"result": {"points": [point]}})
    res = search_points(client, "col", [0.1], 5)
    assert len(res) == 1
    assert res[0].id == "2"
    assert res[0].payload["b"] == 2
