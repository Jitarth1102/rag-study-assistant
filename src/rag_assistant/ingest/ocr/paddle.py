"""PaddleOCR wrapper with version-tolerant kwargs handling."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict, Any

from rag_assistant.ingest.ocr.normalize import normalize_ocr_result

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - dependency loading guard
    PaddleOCR = None  # type: ignore


class OCRIncompatibleError(RuntimeError):
    """Raised when PaddleOCR API is incompatible at runtime."""


class PaddleOCREngine:
    def __init__(self, lang: str = "en"):
        if PaddleOCR is None:
            raise ImportError("paddleocr is not installed or failed to load")
        kwargs: dict[str, Any] = {"lang": lang, "use_angle_cls": False}
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            if "show_log" in sig.parameters:
                kwargs["show_log"] = False
            if "use_angle_cls" not in sig.parameters:
                kwargs.pop("use_angle_cls", None)
            self.ocr = PaddleOCR(**kwargs)
        except TypeError as exc:
            # Retry with minimal kwargs
            try:
                self.ocr = PaddleOCR(lang=lang)
            except Exception:
                raise OCRIncompatibleError(f"PaddleOCR init failed: {exc}") from exc
        except Exception as exc:
            raise OCRIncompatibleError(f"PaddleOCR init failed: {exc}") from exc

    def ocr_page(self, image_path: str, page_num: int) -> Dict[str, Any]:
        try:
            result = self.ocr.ocr(image_path)
        except TypeError as exc:
            raise OCRIncompatibleError("PaddleOCR API mismatch", exc) from exc
        normalized = normalize_ocr_result(result, page_num)
        return normalized


def save_ocr_json(ocr_json: dict, asset_id: str, ocr_dir: Path, page_num: int) -> Path:
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_path = ocr_dir / f"page_{page_num:04d}.json"
    out_path.write_text(json.dumps(ocr_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def analyze_ocr_stats(ocr_json: dict) -> dict:
    text_len = sum(len(block.get("text", "")) for block in ocr_json.get("blocks", []))
    confidences = [block.get("confidence", 0.0) for block in ocr_json.get("blocks", [])]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    needs_caption = 1 if text_len < 80 else 0
    return {"text_len": text_len, "avg_conf": avg_conf, "needs_caption": needs_caption}


__all__ = ["PaddleOCREngine", "save_ocr_json", "analyze_ocr_stats"]
