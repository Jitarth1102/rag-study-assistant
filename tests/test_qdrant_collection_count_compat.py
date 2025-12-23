from rag_assistant.retrieval.vector_store.qdrant import QdrantStore


class DummyInfoPoints:
    def __init__(self):
        self.points_count = 5


class DummyInfoVectors:
    def __init__(self):
        self.vectors_count = 7


class DummyClient:
    def __init__(self, info):
        self._info = info

    def get_collection(self, name):
        return self._info

    def get_collections(self):
        class C:
            collections = []

        return C()


def test_collection_count_points(monkeypatch):
    store = QdrantStore()
    monkeypatch.setattr(store, "client", DummyClient(DummyInfoPoints()))
    assert store.get_collection_point_count() == 5


def test_collection_count_vectors(monkeypatch):
    store = QdrantStore()
    monkeypatch.setattr(store, "client", DummyClient(DummyInfoVectors()))
    assert store.get_collection_point_count() == 7


def test_collection_count_unknown(monkeypatch):
    store = QdrantStore()
    monkeypatch.setattr(store, "client", DummyClient(object()))
    assert store.get_collection_point_count() == 0
