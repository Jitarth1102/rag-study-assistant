"""Chat service stub."""

from __future__ import annotations

from rag_assistant.domain.models import ChatResponse, Citation


def ask(subject_id: str, question: str) -> dict:
    response = ChatResponse(
        answer=f"Placeholder answer for subject '{subject_id}': {question}",
        citations=[Citation(source="placeholder.pdf", page=1, quote="Example citation")],
    )
    return {
        "answer": response.answer,
        "citations": [citation.__dict__ for citation in response.citations],
    }
