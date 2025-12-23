"""OCR self-test utility."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from rag_assistant.config import load_config
from rag_assistant.ingest.ocr.factory import get_ocr_engine


def run_ocr_selftest(config=None) -> dict:
    cfg = config or load_config()
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "OCR TEST 123", fill=(0, 0, 0))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        image_path = tmp.name

    engine, warning, engine_name = get_ocr_engine(lang=cfg.ingest.ocr_lang, config=cfg)
    ocr_json = engine.ocr_page(image_path, 1) if hasattr(engine, "ocr_page") else {"blocks": []}
    text = "\n".join(b.get("text", "") for b in ocr_json.get("blocks", []))
    try:
        Path(image_path).unlink(missing_ok=True)
    except Exception:
        pass
    return {
        "engine": engine_name,
        "warning": warning,
        "text": text,
        "blocks": len(ocr_json.get("blocks", [])),
    }


__all__ = ["run_ocr_selftest"]
