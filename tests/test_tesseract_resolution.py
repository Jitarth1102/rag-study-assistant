import types
import subprocess

from rag_assistant.ingest.ocr import tesseract as tess_mod
from rag_assistant.config import load_config


def test_resolve_tesseract_prefers_config(monkeypatch):
    cfg = load_config()
    cfg.ingest.tesseract_cmd = "/custom/tess"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=b"", stderr=b""))
    monkeypatch.setattr(tess_mod.pytesseract.pytesseract, "tesseract_cmd", "")
    engine = tess_mod.TesseractOCREngine(lang="en", config=cfg)
    assert engine.tesseract_cmd == "/custom/tess"


def test_resolve_tesseract_which(monkeypatch):
    cfg = load_config()
    cfg.ingest.tesseract_cmd = ""
    monkeypatch.setattr(tess_mod.shutil, "which", lambda name: "/usr/bin/tess")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=b"", stderr=b""))
    monkeypatch.setattr(tess_mod.pytesseract.pytesseract, "tesseract_cmd", "")
    engine = tess_mod.TesseractOCREngine(lang="en", config=cfg)
    assert engine.tesseract_cmd == "/usr/bin/tess"
