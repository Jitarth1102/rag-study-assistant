from types import SimpleNamespace

from rag_assistant.rag import answerer


def test_query_embedding_normalized_to_list(monkeypatch):
    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts):
            # Simulate numpy array object
            return [SimpleNamespace(tolist=lambda: [0.1] * 384)]

    called = {}

    class DummyStore:
        def __init__(self):
            pass

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            called["vector"] = vector
            return []

        def search_notes(self, vector, subject_id, limit):
            return []

    monkeypatch.setattr(answerer, "Embedder", lambda *args, **kwargs: DummyEmbedder())
    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())

    answerer.ask("subj", "q", 3)
    assert isinstance(called["vector"], list)
    assert all(isinstance(x, float) for x in called["vector"])
