"""Chat service using RAG answerer."""

from __future__ import annotations

from rag_assistant.config import load_config
from rag_assistant.rag.answerer import ask as rag_ask


def ask(subject_id: str, question: str) -> dict:
    cfg = load_config()
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
