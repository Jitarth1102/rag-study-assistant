from rag_assistant.rag import answerer


def test_chat_empty_index(monkeypatch):
    class DummyStore:
        def get_collection_point_count(self):
            return 0

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.1]]

    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "Embedder", lambda *args, **kwargs: DummyEmbedder())
    res = answerer.ask("subj", "question", 5)
    assert "I don’t have any indexed content" in res["answer"]
