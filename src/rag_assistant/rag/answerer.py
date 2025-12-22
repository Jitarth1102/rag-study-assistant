"""Notes-only QA over indexed chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from jinja2 import Template

from rag_assistant.config import load_config
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore
from rag_assistant.llm.provider import generate_answer


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "llm" / "prompts" / "answer.md"
    return prompt_path.read_text(encoding="utf-8")


def _format_context(chunks: List[dict]) -> str:
    lines = []
    for chunk in chunks:
        snippet = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id")
        source = chunk.get("source") or chunk.get("asset_id")
        page = chunk.get("page_num")
        lines.append(f"[chunk:{chunk_id}] (asset={source}, page={page})\n{snippet}\n")
    return "\n".join(lines)


def ask(subject_id: str, question: str, top_k: int, config=None) -> dict:
    cfg = config or load_config()
    embedder = Embedder()
    store = QdrantStore()

    query_emb = embedder.embed_texts([question])[0]
    hits = store.search(query_emb, subject_id=subject_id, limit=top_k)
    if not hits:
        return {"answer": "Answer not found in your notes.", "citations": []}

    prompt_template = Template(_load_prompt())
    context = _format_context(hits)
    prompt = prompt_template.render(context=context, question=question)

    try:
        answer_text = generate_answer(prompt, cfg)
    except Exception as exc:  # pragma: no cover
        answer_text = f"LLM error: {exc}"

    citations = []
    for hit in hits:
        citations.append(
            {
                "asset_id": hit.get("asset_id"),
                "filename": hit.get("source"),
                "page": hit.get("page_num"),
                "chunk_id": hit.get("chunk_id"),
                "quote": hit.get("text", "")[:240],
                "image_path": hit.get("image_path"),
            }
        )
    return {"answer": answer_text, "citations": citations}


__all__ = ["ask"]
