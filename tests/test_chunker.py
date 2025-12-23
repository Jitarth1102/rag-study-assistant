from rag_assistant.ingest.chunking.layout_chunker import chunk_ocr_blocks


def test_chunker_deterministic():
    ocr = {
        "blocks": [
            {"text": "A", "bbox": [0, 0, 10, 10]},
            {"text": "B", "bbox": [0, 20, 10, 30]},
            {"text": "C", "bbox": [0, 40, 10, 50]},
        ]
    }
    chunks1 = chunk_ocr_blocks("s1", "asset", 1, ocr, max_chunk_chars=5, min_chunk_chars=1, overlap_blocks=1)
    chunks2 = chunk_ocr_blocks("s1", "asset", 1, ocr, max_chunk_chars=5, min_chunk_chars=1, overlap_blocks=1)
    assert [c["chunk_id"] for c in chunks1] == [c["chunk_id"] for c in chunks2]
    assert len(chunks1) == len(chunks2)
