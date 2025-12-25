"""Rule-based judge to decide if web search should be used."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List

from rag_assistant.config import load_config


@dataclass
class JudgeDecision:
    do_search: bool
    reason: str
    suggested_queries: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _looks_like_definition(question: str) -> bool:
    q = question.lower()
    patterns = ["what is", "define", "explain", "describe", "derivation", "proof", "how does", "why is"]
    return any(p in q for p in patterns)


def should_search_web(
    question: str,
    rag_hits: List[dict],
    rag_debug: dict | None = None,
    config=None,
    force_even_if_rag_strong: bool = False,
) -> JudgeDecision:
    cfg = config or load_config()
    web_cfg = cfg.web
    if not getattr(web_cfg, "enabled", False):
        return JudgeDecision(do_search=False, reason="web_disabled", suggested_queries=[])

    hit_count = rag_debug.get("hit_count_after_filter", 0) if rag_debug else len(rag_hits)
    top_score = 0.0
    if rag_hits:
        top_score = rag_hits[0].get("score") or rag_debug.get("top_hits_preview", [{}])[0].get("score", 0) if rag_debug else 0
        if isinstance(top_score, str):
            try:
                top_score = float(top_score)
            except Exception:
                top_score = 0.0
    min_hits = getattr(web_cfg, "min_rag_hits_to_skip_web", 3)
    min_score = getattr(web_cfg, "min_rag_score_to_skip_web", 0.65)
    hits_gate = min_hits is not None and min_hits > 0
    score_gate = min_score is not None and min_score > 0
    skip_by_hits = hits_gate and hit_count >= min_hits
    skip_by_score = score_gate and top_score is not None and top_score >= min_score
    if (skip_by_hits or skip_by_score) and not force_even_if_rag_strong:
        return JudgeDecision(do_search=False, reason="rag_confident", suggested_queries=[])

    if force_even_if_rag_strong:
        return JudgeDecision(do_search=True, reason="forced_by_user", suggested_queries=[question])

    # If context is weak and question is definitional/explanatory, try web.
    if _looks_like_definition(question):
        return JudgeDecision(do_search=True, reason="definition_with_weak_rag", suggested_queries=[question])

    # If no hits at all, try web with the original question.
    if hit_count == 0:
        return JudgeDecision(do_search=True, reason="no_hits", suggested_queries=[question])

    # Default: do not search.
    return JudgeDecision(do_search=False, reason="default_no_search", suggested_queries=[])


__all__ = ["JudgeDecision", "should_search_web"]
