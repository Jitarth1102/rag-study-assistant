"""Utilities to sanitize text and build concise web search queries."""

from __future__ import annotations

import re
from typing import List, Optional


CRITIQUE_STOP_WORDS = {"weak", "lacking", "suggestions", "abrupt", "critique", "issue", "missing", "expand", "section"}
FILENAME_EXTS = {".pdf", ".pptx", ".ppt", ".doc", ".docx"}


def sanitize_text(text: str) -> str:
    """Remove markdown, code fences, and filenames/extensions from queries."""
    if not text:
        return ""
    cleaned = re.sub(r"[`*_#>\\$]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for ext in FILENAME_EXTS:
        cleaned = cleaned.replace(ext, " ")
    tokens = [t for t in cleaned.split() if t.lower() not in CRITIQUE_STOP_WORDS]
    return " ".join(tokens)


def _clip_words(text: str, max_words: int = 12) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def build_web_queries(
    *,
    subject: Optional[str] = None,
    asset_title: Optional[str] = None,
    intents: Optional[List[str]] = None,
    question: Optional[str] = None,
    max_queries: int = 3,
) -> List[str]:
    candidates: List[str] = []
    for intent in intents or []:
        cleaned = sanitize_text(intent)
        if cleaned:
            candidates.append(cleaned)
    if question:
        cleaned_q = sanitize_text(question)
        if cleaned_q:
            candidates.append(cleaned_q)
    if not candidates and asset_title:
        candidates.append(sanitize_text(asset_title))
    if not candidates and subject:
        candidates.append(sanitize_text(subject))
    clipped = [_clip_words(c, 12) for c in candidates if c]
    deduped = []
    seen = set()
    for q in clipped:
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
        if len(deduped) >= max_queries:
            break
    return deduped


def build_queries_for_gap(gap, subject_hint: Optional[str], max_queries: int = 3) -> List[str]:
    intents = [gap.what_to_add] if getattr(gap, "what_to_add", None) else []
    return build_web_queries(
        subject=subject_hint,
        asset_title=getattr(gap, "section_title", None),
        intents=intents,
        question=None,
        max_queries=max_queries,
    )


__all__ = ["sanitize_text", "build_web_queries", "build_queries_for_gap"]
