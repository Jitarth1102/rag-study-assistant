from rag_assistant.ingest.ocr.normalize import normalize_ocr_result


def test_paddle_tuple_shape():
    raw = [[[[0, 0], [10, 0], [10, 10], [0, 10]], ("hello", 0.9)]]
    out = normalize_ocr_result(raw, 1)
    assert out["blocks"][0]["text"] == "hello"
    assert out["blocks"][0]["confidence"] == 0.9


def test_paddle_triple_shape():
    raw = [[[[0, 0], [10, 0], [10, 10], [0, 10]], "world", 0.8]]
    out = normalize_ocr_result(raw, 1)
    assert out["blocks"][0]["text"] == "world"


def test_flat_string():
    out = normalize_ocr_result("simple text", 1)
    assert out["blocks"][0]["text"] == "simple text"


def test_empty_results():
    out = normalize_ocr_result([], 1)
    assert out["blocks"] == []


def test_malformed_safe():
    out = normalize_ocr_result(123, 1)
    assert out["blocks"] == []
