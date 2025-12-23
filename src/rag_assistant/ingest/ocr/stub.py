"""Stub OCR engine for fallback."""

from __future__ import annotations

from typing import Dict

from rag_assistant.ingest.ocr.normalize import normalize_ocr_result


class StubOCREngine:
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def ocr_page(self, image_path: str, page_num: int) -> Dict:
        return normalize_ocr_result(
            {
                "blocks": [
                    {"text": "(OCR stub — no OCR engine available)", "bbox": [0, 0, 1, 1], "confidence": 0.0}
                ]
            },
            page_num,
        )


__all__ = ["StubOCREngine"]
