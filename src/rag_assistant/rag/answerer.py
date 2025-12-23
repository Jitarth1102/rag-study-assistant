"""Notes-only QA over indexed chunks."""

from __future__ import annotations

from pathlib import Path
from typing import List

from jinja2 import Template

from rag_assistant.config import load_config
from rag_assistant.llm.provider import generate_answer
from rag_assistant.retrieval.context_expander import expand_with_neighbors
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore


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
    embedder = Embedder(config=cfg)
    store = QdrantStore()

    point_count = 0
    try:
        point_count = store.get_collection_point_count()
    except Exception:
        point_count = 0

    if point_count == 0:
        return {
            "answer": "I don’t have any indexed content for this subject yet. Go to Upload → Index new uploads. If OCR blocks are empty, install/enable Tesseract OCR.",
            "citations": [],
        }

    query_embs = embedder.embed_texts([question])
    if not query_embs:
        return {
            "answer": "I don’t have any indexed content for this subject yet. Go to Upload → Index new uploads. If OCR blocks are empty, install/enable Tesseract OCR.",
            "citations": [],
        }
    query_emb = query_embs[0]
    hits = store.search(query_emb, subject_id=subject_id, limit=top_k)
    if not hits:
        return {"answer": "Answer not found in your notes.", "citations": []}

    try:
        expanded = expand_with_neighbors(
            hits,
            window=getattr(cfg.retrieval, "neighbor_window", 1),
            max_extra=getattr(cfg.retrieval, "max_neighbor_chunks", 12),
            db_path=Path(cfg.database.sqlite_path),
        )
    except Exception:
        expanded = hits
    extra_neighbors = max(0, len(expanded) - len(hits))
    context_chunks = expanded

    prompt_template = Template(_load_prompt())
    context = _format_context(context_chunks)
    prompt = prompt_template.render(context=context, question=question)

    try:
        answer_text = generate_answer(prompt, cfg)
    except Exception as exc:  # pragma: no cover
        answer_text = f"LLM error: {exc}"

    source_lookup = {}
    for hit in hits:
        aid = hit.get("asset_id")
        if aid and hit.get("source"):
            source_lookup[aid] = hit.get("source")

    citations = []
    seen_cids = set()
    for chunk in context_chunks:
        cid = chunk.get("chunk_id") or f"{chunk.get('asset_id')}:{chunk.get('page_num')}:{chunk.get('start_block')}"
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        asset_id = chunk.get("asset_id")
        citations.append(
            {
                "asset_id": asset_id,
                "filename": chunk.get("source") or source_lookup.get(asset_id) or asset_id,
                "page": chunk.get("page_num"),
                "chunk_id": cid,
                "quote": chunk.get("text", "")[:240],
                "image_path": chunk.get("image_path"),
            }
        )
    return {
        "answer": answer_text,
        "citations": citations,
        "context_expanded": extra_neighbors,
        "context_chunks": len(context_chunks),
    }


__all__ = ["ask"]
