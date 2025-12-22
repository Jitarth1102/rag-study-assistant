"""Placeholder Tesseract OCR engine."""

from __future__ import annotations

from typing import Dict

from rag_assistant.ingest.ocr.normalize import normalize_ocr_result


class TesseractOCREngine:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        # Not implementing real tesseract wiring here; acts as a stub fallback.

    def ocr_page(self, image_path: str, page_num: int) -> Dict:
        # In a real implementation, run pytesseract here.
        return normalize_ocr_result("", page_num)


__all__ = ["TesseractOCREngine"]
