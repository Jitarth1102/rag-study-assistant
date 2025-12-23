"""Factory for OCR engines with PaddleOCR fallback handling."""

from __future__ import annotations

import logging

from rag_assistant.config import load_config
from rag_assistant.ingest.ocr.paddle import PaddleOCREngine, OCRIncompatibleError
from rag_assistant.ingest.ocr.tesseract import TesseractOCREngine
from rag_assistant.ingest.ocr.stub import StubOCREngine

logger = logging.getLogger(__name__)


def get_ocr_engine(lang: str = "en", config=None):
    """Return OCR engine, warning (if any), and engine name."""
    cfg = config or load_config()
    choice = cfg.ingest.ocr_engine.lower()
    if choice == "tesseract":
        engine = TesseractOCREngine(lang=lang, config=cfg)
        logger.info("OCR engine selected", extra={"engine": "tesseract", "tesseract_cmd": engine.tesseract_cmd})
        return engine, None, "tesseract"

    if choice == "paddle":
        engine = PaddleOCREngine(lang=lang)
        logger.info("OCR engine selected", extra={"engine": "paddle"})
        return engine, None, "paddle"

    # auto: try paddle -> tesseract -> stub
    try:
        engine = PaddleOCREngine(lang=lang)
        logger.info("OCR engine selected", extra={"engine": "paddle"})
        return engine, None, "paddle"
    except Exception as exc:
        warning = f"PaddleOCR unavailable or incompatible: {exc}. Using fallback OCR."
        logger.warning(warning)
        try:
            engine = TesseractOCREngine(lang=lang, config=cfg)
            logger.info("OCR engine selected", extra={"engine": "tesseract", "tesseract_cmd": engine.tesseract_cmd})
            return engine, warning, "tesseract"
        except Exception as texc:
            stub_warning = warning + f" Tesseract failed: {texc}. Using stub OCR."
            logger.warning(stub_warning)
            return StubOCREngine(lang=lang), stub_warning, "stub"


__all__ = ["get_ocr_engine", "StubOCREngine"]
