"""Utilities to sanitize text and build concise web search queries."""

from __future__ import annotations

import re
from typing import List, Optional


CRITIQUE_STOP_WORDS = {"weak", "lacking", "suggestions", "abrupt", "critique", "issue", "missing", "expand", "section"}
FILENAME_EXTS = {".pdf", ".pptx", ".ppt", ".doc", ".docx"}
STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "a",
    "an",
    "is",
    "are",
    "this",
    "that",
    "these",
    "those",
    "as",
    "by",
    "from",
    "at",
    "be",
    "it",
    "its",
}
RL_ANCHORS = [
    "mdp",
    "policy",
    "value function",
    "q function",
    "bellman equation",
    "q-learning",
    "exploration",
    "exploitation",
    "dqn",
]


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


def _extract_keywords(text: str, limit: int = 10) -> List[str]:
    tokens = re.findall(r"[A-Za-z][\w'-]+", text.lower())
    cleaned = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and t not in CRITIQUE_STOP_WORDS]
    freq = {}
    for tok in cleaned:
        freq[tok] = freq.get(tok, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in sorted_terms[:limit]]


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


def build_section_queries(section_title: str, section_text: str, slide_context: str, max_queries: int = 3) -> List[str]:
    """Build concise, section-grounded queries."""
    title_terms = _extract_keywords(section_title or "", limit=4)
    section_terms = _extract_keywords(section_text or "", limit=6)
    slide_terms = _extract_keywords(slide_context or "", limit=10)
    anchors: List[str] = []
    for term in title_terms + section_terms:
        if term not in anchors:
            anchors.append(term)
    for term in slide_terms:
        if term not in anchors:
            anchors.append(term)
    for anchor in RL_ANCHORS:
        if anchor not in anchors:
            anchors.append(anchor)
    anchors = anchors[:10]
    base_query = sanitize_text(" ".join(anchors))
    queries: List[str] = []
    if base_query:
        queries.append(_clip_words(base_query, 16))
    extra_terms = []
    if "policy" in anchors and "mdp" in anchors:
        extra_terms = ["policy mdp definition bellman equation value function"]
    elif "q-learning" in anchors or "q" in anchors:
        extra_terms = ["q-learning update rule alpha gamma max action explanation"]
    for term in extra_terms:
        q = _clip_words(sanitize_text(term), 18)
        if q:
            queries.append(q)
    deduped: List[str] = []
    seen = set()
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q[:120])
        if len(deduped) >= max_queries:
            break
    return deduped


__all__ = ["sanitize_text", "build_web_queries", "build_queries_for_gap", "build_section_queries"]
