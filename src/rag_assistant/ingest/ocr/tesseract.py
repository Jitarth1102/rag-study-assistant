"""Tesseract OCR engine with path resolution."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytesseract
from PIL import Image

from rag_assistant.config import load_config
from rag_assistant.ingest.ocr.normalize import normalize_ocr_result


COMMON_PATHS = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]


def _resolve_tesseract_cmd(config) -> str:
    if config.ingest.tesseract_cmd:
        return config.ingest.tesseract_cmd
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in COMMON_PATHS:
        if Path(candidate).exists():
            return candidate
    return ""


def _normalize_lang(lang: str) -> str:
    return "eng" if lang.lower() == "en" else lang


class TesseractOCREngine:
    def __init__(self, lang: str = "en", config=None):
        self.cfg = config or load_config()
        self.lang = _normalize_lang(lang)
        cmd = _resolve_tesseract_cmd(self.cfg)
        if not cmd:
            raise RuntimeError("tesseract binary not found. Set ingest.tesseract_cmd or ensure it is on PATH.")
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.tessdata_dir = self._resolve_tessdata_dir(cmd, self.lang)
        try:
            subprocess.run([cmd, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, env=self._build_env())
        except Exception as exc:
            raise RuntimeError(f"tesseract not runnable at {cmd}: {exc}") from exc
        self.tesseract_cmd = cmd
        os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

    def ocr_page(self, image_path, page_num: int) -> Dict:
        try:
            env = self._build_env()
            text = pytesseract.image_to_string(str(image_path), lang=self.lang)
        except Exception as exc:
            raise RuntimeError(
                f"Tesseract OCR failed: {exc}. cmd={self.tesseract_cmd}, tessdata_dir={self.tessdata_dir}, lang={self.lang}. "
                f"Ensure {self.lang}.traineddata exists or set ingest.tessdata_dir."
            ) from exc
        return normalize_ocr_result(text, page_num)

    def _build_env(self):
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = str(self.tessdata_dir)
        return env

    def _resolve_tessdata_dir(self, cmd: str, lang: str) -> Path:
        cfg_dir = self.cfg.ingest.tessdata_dir
        candidates = []
        if cfg_dir:
            candidates.append(Path(cfg_dir))
        cmd_path = Path(cmd)
        if cmd_path.parent.name == "bin":
            candidates.append(cmd_path.parent.parent / "share" / "tessdata")
        candidates.extend([Path("/opt/homebrew/share/tessdata"), Path("/usr/local/share/tessdata")])
        for cand in candidates:
            lang_file = cand / f"{lang}.traineddata"
            if lang_file.exists():
                return cand
        raise RuntimeError(f"tessdata for lang '{lang}' not found in candidates: {candidates}")


__all__ = ["TesseractOCREngine", "_resolve_tesseract_cmd"]
