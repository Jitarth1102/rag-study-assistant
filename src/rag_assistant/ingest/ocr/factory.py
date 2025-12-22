"""Factory for OCR engines with PaddleOCR fallback handling."""

from __future__ import annotations

import logging

from rag_assistant.ingest.ocr.paddle import PaddleOCREngine, OCRIncompatibleError
from rag_assistant.ingest.ocr.tesseract import TesseractOCREngine
from rag_assistant.ingest.ocr.stub import StubOCREngine

logger = logging.getLogger(__name__)


def get_ocr_engine(lang: str = "en"):
    """Return OCR engine and optional warning message if fallback used."""
    try:
        engine = PaddleOCREngine(lang=lang)
        return engine, None
    except (ImportError, OCRIncompatibleError) as exc:
        warning = f"PaddleOCR unavailable or incompatible: {exc}. Using fallback OCR."
        logger.warning(warning)
        try:
            engine = TesseractOCREngine(lang=lang)
            return engine, warning
        except Exception:
            stub_warning = warning + " Tesseract not available; using stub OCR."
            return StubOCREngine(lang=lang), stub_warning


__all__ = ["get_ocr_engine", "StubOCREngine"]
