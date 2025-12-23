import types
import pytest

from rag_assistant.ingest.ocr import paddle as paddle_module


class DummyOCRWithShowLog:
    def __init__(self, lang="en", use_angle_cls=False, show_log=True):
        self.args = {"lang": lang, "use_angle_cls": use_angle_cls, "show_log": show_log}

    def ocr(self, *args, **kwargs):  # pragma: no cover - not used
        return []


class DummyOCRNoShowLog:
    def __init__(self, lang="en", use_angle_cls=False):
        self.args = {"lang": lang, "use_angle_cls": use_angle_cls}

    def ocr(self, *args, **kwargs):  # pragma: no cover - not used
        return []


def test_paddle_wrapper_with_show_log(monkeypatch):
    monkeypatch.setattr(paddle_module, "PaddleOCR", DummyOCRWithShowLog)
    engine = paddle_module.PaddleOCREngine(lang="en")
    assert isinstance(engine.ocr, DummyOCRWithShowLog)
    assert engine.ocr.args.get("show_log") is False


def test_paddle_wrapper_without_show_log(monkeypatch):
    monkeypatch.setattr(paddle_module, "PaddleOCR", DummyOCRNoShowLog)
    engine = paddle_module.PaddleOCREngine(lang="en")
    assert isinstance(engine.ocr, DummyOCRNoShowLog)
    assert "show_log" not in engine.ocr.args
