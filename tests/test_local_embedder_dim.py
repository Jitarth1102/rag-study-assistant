from rag_assistant.retrieval.embedder import Embedder, get_embedding_dim


class DummyModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return [[0.1] * 384 for _ in texts]


def test_local_embedder_dimension(monkeypatch):
    import rag_assistant.retrieval.embedder as emb

    monkeypatch.setattr(emb, "SentenceTransformer", lambda name: DummyModel())
    emb._LOCAL_MODEL_CACHE.clear()
    embedder = Embedder(config=None)
    embedder.provider = "local"
    embedder.local_model_name = "dummy"
    vecs = embedder.embed_texts(["hello"])
    assert len(vecs[0]) == 384
    assert get_embedding_dim() == 384
