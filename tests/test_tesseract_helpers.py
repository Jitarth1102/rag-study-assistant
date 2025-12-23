import types
import pytest

from rag_assistant.ingest.ocr import tesseract as tess_mod
from rag_assistant.ingest.ocr import factory
from rag_assistant.config import load_config


def test_normalize_lang():
    assert tess_mod._normalize_lang("en") == "eng"
    assert tess_mod._normalize_lang("eng") == "eng"


def test_resolve_tessdata_dir_prefers_common(monkeypatch):
    cfg = load_config()
    cfg.ingest.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    cfg.ingest.ocr_lang = "en"

    class FakePath:
        def __init__(self, path: str):
            self.path = path

        def __truediv__(self, other):
            return FakePath(f"{self.path}/{other}")

        @property
        def parent(self):
            parts = self.path.rstrip('/').split('/')
            return FakePath('/'.join(parts[:-1]) if len(parts) > 1 else '/')

        @property
        def name(self):
            return self.path.rstrip('/').split('/')[-1]

        def exists(self):
            return self.path.endswith("eng.traineddata") or self.path.endswith("tessdata")

        def __str__(self):
            return self.path

    monkeypatch.setattr(tess_mod, "Path", FakePath)
    monkeypatch.setattr(tess_mod.shutil, "which", lambda name: cfg.ingest.tesseract_cmd)
    monkeypatch.setattr(tess_mod.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=b"", stderr=b""))
    engine = tess_mod.TesseractOCREngine(lang="en", config=cfg)
    assert "tessdata" in str(engine.tessdata_dir)
    assert engine.lang == "eng"


def test_factory_no_fallback_for_explicit_tesseract(monkeypatch):
    cfg = load_config()
    cfg.ingest.ocr_engine = "tesseract"

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(factory, "TesseractOCREngine", boom)
    with pytest.raises(RuntimeError):
        factory.get_ocr_engine(lang="eng", config=cfg)
