import pathlib

from rag_assistant.ingest.ocr import selftest


def test_selftest_uses_file(monkeypatch, tmp_path):
    called = {}

    class DummyEngine:
        def __init__(self, *a, **k):
            pass

        def ocr_page(self, path, page_num):
            called["path"] = path
            return {"blocks": [{"text": "OCR TEST 123"}]}

    def fake_get_engine(lang="eng", config=None):
        return DummyEngine(), None, "dummy"

    monkeypatch.setattr(selftest, "get_ocr_engine", fake_get_engine)
    res = selftest.run_ocr_selftest()
    assert "path" in called
    assert isinstance(called["path"], (str, pathlib.Path))
    assert "OCR TEST 123" in res["text"]
