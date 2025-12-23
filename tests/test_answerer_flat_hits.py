import pytest

from rag_assistant.rag import answerer


def test_answerer_handles_flat_hits(monkeypatch):
    class DummyStore:
        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return [
                {
                    "chunk_id": "chunk1",
                    "text": "Instructor: Prof. Boqing Gong",
                    "page_num": 2,
                    "asset_id": "asset1",
                    "source": "slides.pdf",
                    "image_path": "/tmp/img.png",
                    "score": 0.9,
                }
            ]

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts):
            return [[0.1] * 384]

    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "Embedder", lambda *args, **kwargs: DummyEmbedder())
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "Prof. Boqing Gong")
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda chunks, **kwargs: chunks)

    res = answerer.ask("machine-learning", "Who is the instructor?", top_k=3)
    assert "Prof. Boqing Gong" in res["answer"]
    assert res["citations"]
    cite = res["citations"][0]
    assert cite["chunk_id"] == "chunk1"
    assert cite["page"] == 2
    assert cite["filename"] == "slides.pdf"
    assert res["debug"]["hit_count_raw"] == 1


def test_answerer_handles_nested_payload(monkeypatch):
    class DummyStore:
        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return [
                {
                    "payload": {
                        "chunk_id": "chunk2",
                        "text": "Instructor: Prof. Boqing Gong",
                        "page_num": 2,
                        "asset_id": "asset1",
                        "source": "slides.pdf",
                        "image_path": "/tmp/img.png",
                        "score": 0.9,
                    }
                }
            ]

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts):
            return [[0.1] * 384]

    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "Embedder", lambda *args, **kwargs: DummyEmbedder())
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "Prof. Boqing Gong")
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda chunks, **kwargs: chunks)

    res = answerer.ask("machine-learning", "Who is the instructor?", top_k=3)
    assert "Prof. Boqing Gong" in res["answer"]
    assert res["citations"]
    cite = res["citations"][0]
    assert cite["chunk_id"] == "chunk2"
    assert cite["page"] == 2
