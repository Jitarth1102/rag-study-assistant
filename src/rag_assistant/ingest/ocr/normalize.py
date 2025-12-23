"""Normalize OCR outputs into a consistent schema."""

from __future__ import annotations

from typing import Any, Dict, List


def _default_bbox() -> list:
    return [0, 0, 1, 1]


def normalize_ocr_result(raw_result: Any, page_num: int) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []

    # Case: already normalized dict
    if isinstance(raw_result, dict) and "blocks" in raw_result:
        for block in raw_result.get("blocks", []):
            text = (block.get("text") or "").strip()
            if not text:
                continue
            bbox = block.get("bbox") or _default_bbox()
            conf = float(block.get("confidence", 0.0))
            blocks.append({"text": text, "bbox": bbox, "confidence": conf})
        return {"page": page_num, "blocks": blocks, "width": raw_result.get("width", 0), "height": raw_result.get("height", 0)}

    # Case: string result
    if isinstance(raw_result, str):
        text = raw_result.strip()
        if text:
            blocks.append({"text": text, "bbox": _default_bbox(), "confidence": 0.0})
        return {"page": page_num, "blocks": blocks, "width": 0, "height": 0}

    # Case: list-based results (Paddle styles)
    if isinstance(raw_result, list):
        lines = raw_result
        for line in lines:
            text = ""
            conf = 0.0
            bbox = _default_bbox()
            if isinstance(line, (list, tuple)):
                if len(line) >= 2 and isinstance(line[0], (list, tuple)):
                    bbox_candidate = line[0]
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = str(line[1][0])
                        try:
                            conf = float(line[1][1])
                        except Exception:
                            conf = 0.0
                    elif len(line) >= 3:
                        text = str(line[1])
                        try:
                            conf = float(line[2])
                        except Exception:
                            conf = 0.0
                    else:
                        text = str(line[1]) if len(line) > 1 else ""
                    try:
                        xs = [pt[0] for pt in bbox_candidate]
                        ys = [pt[1] for pt in bbox_candidate]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    except Exception:
                        bbox = _default_bbox()
                elif len(line) >= 2:
                    # (text, conf)
                    text = str(line[0])
                    try:
                        conf = float(line[1])
                    except Exception:
                        conf = 0.0
            text = text.strip()
            if not text:
                continue
            blocks.append({"text": text, "bbox": bbox, "confidence": conf})
        return {"page": page_num, "blocks": blocks, "width": 0, "height": 0}

    # Fallback: unknown format
    return {"page": page_num, "blocks": blocks, "width": 0, "height": 0}


__all__ = ["normalize_ocr_result"]
