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
        raise TypeError("__init__() got an unexpected keyword argument 'cls'")


def test_paddle_wrapper_does_not_pass_cls(monkeypatch):
    monkeypatch.setattr(paddle_module, "PaddleOCR", RejectPredictOCR)
    engine = paddle_module.PaddleOCREngine(lang="en")
    out = engine.ocr_page("/tmp/fake", 1)
    assert "blocks" in out


def test_factory_fallback_on_incompatible(monkeypatch):
    monkeypatch.setattr(paddle_module, "PaddleOCR", BadOCR)
    engine, warning = get_ocr_engine()
    # factory should at least return a fallback engine (tesseract or stub)
    assert warning is not None
