from rag_assistant.rag import answerer


def test_retry_without_subject_filter(monkeypatch):
    class DummyStore:
        def __init__(self):
            self.calls = []

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            self.calls.append(subject_id)
            if subject_id:
                return []
            return [
                {
                    "chunk_id": "c1",
                    "text": "hello world",
                    "page_num": 1,
                    "asset_id": "a1",
                    "subject_id": "subj",
                    "score": 0.9,
                    "source": "file.pdf",
                    "image_path": None,
                }
            ]

        def search_notes(self, vector, subject_id, limit):
            return []

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts):
            return [[0.0] * 384]

    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "Embedder", lambda *args, **kwargs: DummyEmbedder())
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "ok")
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda chunks, **kwargs: chunks)

    res = answerer.ask("subject-filter", "q", 5)
    assert res["citations"], "Expected citations from unfiltered search"
    debug = res.get("debug") or {}
    assert debug.get("filter_retried_without_subject") is True
