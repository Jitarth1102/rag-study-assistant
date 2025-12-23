import pytest

from rag_assistant.ingest.ocr import paddle as paddle_module
from rag_assistant.ingest.ocr.factory import get_ocr_engine, StubOCREngine


class RejectPredictOCR:
    def __init__(self, lang="en"):
        self.lang = lang

    def ocr(self, image_path):
        return [([[0, 0], [0, 0], [0, 0], [0, 0]], ("text", 1.0))]


class BadOCR:
    def __init__(self, lang="en"):
        raise RuntimeError("fail init")


def test_paddle_wrapper_does_not_pass_cls(monkeypatch):
    monkeypatch.setattr(paddle_module, "PaddleOCR", RejectPredictOCR)
    engine = paddle_module.PaddleOCREngine(lang="en")
    out = engine.ocr_page("/tmp/fake", 1)
    assert "blocks" in out


def test_factory_fallback_on_incompatible(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    class DummyCfg:
        class Ingest:
            ocr_engine = "auto"
            tesseract_cmd = ""
            tessdata_dir = ""
            ocr_lang = "eng"

        ingest = Ingest()

    monkeypatch.setattr("rag_assistant.ingest.ocr.factory.load_config", lambda: DummyCfg())
    monkeypatch.setattr("rag_assistant.ingest.ocr.factory.PaddleOCREngine", boom)
    monkeypatch.setattr("rag_assistant.ingest.ocr.factory.TesseractOCREngine", lambda *a, **k: StubOCREngine())
    engine, warning, engine_name = get_ocr_engine()
    # factory should at least return a fallback engine (tesseract or stub)
    assert warning is not None
    assert engine_name in {"tesseract", "stub"}
