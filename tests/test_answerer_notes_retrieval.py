from rag_assistant.rag import answerer


def test_answerer_merges_notes_hits(monkeypatch):
    class DummyStore:
        collection = "col"

        def get_collection_point_count(self):
            return 1

        def search(self, vector, subject_id, limit):
            return [
                {
                    "chunk_id": "slide1",
                    "text": "Slide content about topic",
                    "page_num": 1,
                    "asset_id": "asset1",
                    "source": "slides.pdf",
                    "score": 0.9,
                }
            ]

        def search_notes(self, vector, subject_id, limit):
            return [
                {
                    "chunk_id": "note1",
                    "text": "Notes content",
                    "asset_id": "asset1",
                    "notes_id": "notes1",
                    "version": 1,
                    "section_title": "Summary",
                    "source": "asset1",
                    "source_label": "Generated Notes",
                    "source_type": "notes",
                    "score": 0.8,
                }
            ]

    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.1] * 384]

    monkeypatch.setattr(answerer, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(answerer, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(answerer, "generate_answer", lambda prompt, cfg: "answer")
    monkeypatch.setattr(answerer, "expand_with_neighbors", lambda chunks, **kwargs: chunks)

    res = answerer.ask("subj", "question", top_k=3)
    citations = res["citations"]
    assert any(c.get("type") == "notes" for c in citations)
    assert any(c.get("type") == "slide" for c in citations)
    assert any(c.get("source_label") == "Generated Notes" for c in citations if c.get("type") == "notes")
    assert res["debug"]["hit_count_raw"] == 2
