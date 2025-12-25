"""Chat service using RAG answerer."""

from __future__ import annotations

from copy import deepcopy

from rag_assistant.config import load_config
from rag_assistant.rag.answerer import ask as rag_ask


def _apply_web_overrides(cfg, overrides: dict | None):
    if not overrides:
        return cfg
    # shallow copy config to avoid mutating global/default
    cfg_copy = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else deepcopy(cfg)
    web_cfg = getattr(cfg_copy, "web", None)
    if web_cfg is None:
        return cfg_copy
    if "web_enabled_override" in overrides:
        web_cfg.enabled = bool(overrides.get("web_enabled_override"))
    if "web_max_queries_override" in overrides and overrides.get("web_max_queries_override"):
        web_cfg.max_web_queries_per_question = int(overrides["web_max_queries_override"])
    if "web_force_even_if_rag_strong_override" in overrides:
        web_cfg.force_even_if_rag_strong = bool(overrides["web_force_even_if_rag_strong_override"])
    if "web_allowed_domains_override" in overrides and overrides["web_allowed_domains_override"] is not None:
        web_cfg.allowed_domains = list(overrides["web_allowed_domains_override"])
    if "web_blocked_domains_override" in overrides and overrides["web_blocked_domains_override"] is not None:
        web_cfg.blocked_domains = list(overrides["web_blocked_domains_override"])
    return cfg_copy


def ask(subject_id: str, question: str, *, overrides: dict | None = None) -> dict:
    cfg = load_config()
    cfg = _apply_web_overrides(cfg, overrides)
    try:
        return rag_ask(subject_id, question, top_k=cfg.retrieval.top_k, config=cfg)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {
            "answer": (
                "An unexpected error occurred while answering the question. "
                "Please check that the retrieval backend (for example, Qdrant) is running, "
                "that any required API keys are configured, and that the selected subject "
                "has indexed content.\n"
                f"Details: {type(exc).__name__}: {exc}"
            ),
            "citations": [],
        }
