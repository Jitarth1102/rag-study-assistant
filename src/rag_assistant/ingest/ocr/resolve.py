"""Helpers to resolve tesseract paths for diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from rag_assistant.config import load_config
from rag_assistant.ingest.ocr.tesseract import _resolve_tesseract_cmd, _normalize_lang

COMMON_TESSDATA = [
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
]


def resolve_tesseract(config=None) -> Tuple[str, str, str]:
    cfg = config or load_config()
    cmd = _resolve_tesseract_cmd(cfg)
    lang = _normalize_lang(cfg.ingest.ocr_lang)
    tessdata_dir = cfg.ingest.tessdata_dir or ""
    if not tessdata_dir:
        if cmd and Path(cmd).parent.name == "bin":
            tessdata_dir = str(Path(cmd).parent.parent / "share" / "tessdata")
        else:
            for cand in COMMON_TESSDATA:
                if Path(cand).exists():
                    tessdata_dir = cand
                    break
    return cmd, tessdata_dir, lang


__all__ = ["resolve_tesseract"]
