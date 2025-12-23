from types import SimpleNamespace

from rag_assistant.retrieval.vector_store.qdrant import _normalize_points


def test_normalize_points_preserves_chunk_id_from_dict():
    pts = [{"id": "p1", "score": 0.5, "payload": {"chunk_id": "c1", "text": "hi", "page_num": 1}}]
    norm = _normalize_points(pts)
    assert norm[0].payload["chunk_id"] == "c1"
    assert norm[0].payload["text"] == "hi"


def test_normalize_points_preserves_chunk_id_from_object():
    pt = SimpleNamespace(id="p2", score=0.6, payload={"chunk_id": "c2", "text": "hello", "page_num": 2})
    norm = _normalize_points([pt])
    assert norm[0].payload["chunk_id"] == "c2"
    assert norm[0].payload["page_num"] == 2
