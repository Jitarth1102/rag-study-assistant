"""Notes-only QA over indexed chunks."""

from __future__ import annotations

from pathlib import Path
from typing import List
import math

from jinja2 import Template

from rag_assistant.config import load_config
from rag_assistant.llm.provider import generate_answer
from rag_assistant.retrieval.context_expander import expand_with_neighbors
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore
from rag_assistant.retrieval.debug import RetrievalDebug


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "llm" / "prompts" / "answer.md"
    return prompt_path.read_text(encoding="utf-8")


def _get_hit_field(hit: dict, key: str, default=None):
    if key in hit:
        return hit.get(key, default)
    payload = hit.get("payload") if isinstance(hit, dict) else None
    if isinstance(payload, dict) and key in payload:
        return payload.get(key, default)
    return default


def _format_context(chunks: List[dict]) -> str:
    lines = []
    for chunk in chunks:
        snippet = _get_hit_field(chunk, "text", "") or _get_hit_field(chunk, "preview", "")
        chunk_id = _get_hit_field(chunk, "chunk_id")
        source = _get_hit_field(chunk, "source") or _get_hit_field(chunk, "asset_id")
        page = _get_hit_field(chunk, "page_num")
        lines.append(f"[chunk:{chunk_id}] (asset={source}, page={page})\n{snippet}\n")
    return "\n".join(lines)


def _validate_embedding(vec: List[float], expected_dim: int) -> dict:
    if len(vec) != expected_dim:
        raise ValueError(f"Query embedding dim {len(vec)} != expected {expected_dim}")
    has_nan = any(math.isnan(x) or math.isinf(x) for x in vec)
    if has_nan:
        raise ValueError("Query embedding contains NaN/inf")
    vec_min = min(vec) if vec else 0.0
    vec_max = max(vec) if vec else 0.0
    vec_mean = sum(vec) / len(vec) if vec else 0.0
    return {"min": vec_min, "max": vec_max, "mean": vec_mean, "has_nan": has_nan}


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
    # Normalize embedding to plain list[float] for qdrant-client compatibility
    if hasattr(query_emb, "tolist"):
        query_emb = query_emb.tolist()
    query_emb = [float(x) for x in query_emb]
    emb_stats = _validate_embedding(query_emb, expected_dim=cfg.embeddings.vector_size)
    hits = store.search(query_emb, subject_id=subject_id, limit=top_k)
    filter_retried = False
    if not hits:
        hits = store.search(query_emb, subject_id=None, limit=top_k)
        filter_retried = True
    if not hits:
        return {"answer": "Answer not found in your notes.", "citations": [], "debug": RetrievalDebug(
            collection_name=store.collection,
            selected_subject_id=subject_id,
            filter_used={"subject_id": subject_id} if subject_id else None,
            top_k=top_k,
            min_score=getattr(cfg.retrieval, "min_score", 0.0),
            query_embedding_dim=len(query_emb),
            query_embedding_min=emb_stats["min"],
            query_embedding_max=emb_stats["max"],
            query_embedding_mean=emb_stats["mean"],
            query_embedding_has_nan=emb_stats["has_nan"],
            hit_count_raw=0,
            hit_count_after_filter=0,
            top_hits_preview=[],
            filter_retried_without_subject=filter_retried,
        ).to_dict()}

    min_score = getattr(cfg.retrieval, "min_score", 0.0)
    filtered_hits = [h for h in hits if h.get("score") is None or h.get("score", 0) >= min_score]
    if not filtered_hits and hits:
        filtered_hits = hits

    try:
        expanded = expand_with_neighbors(
            filtered_hits,
            window=getattr(cfg.retrieval, "neighbor_window", 1),
            max_extra=getattr(cfg.retrieval, "max_neighbor_chunks", 12),
            db_path=Path(cfg.database.sqlite_path),
        )
    except Exception:
        expanded = filtered_hits
    extra_neighbors = max(0, len(expanded) - len(filtered_hits))
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
        cid = _get_hit_field(chunk, "chunk_id") or f"{_get_hit_field(chunk, 'asset_id')}:{_get_hit_field(chunk, 'page_num')}:{_get_hit_field(chunk, 'start_block')}"
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        asset_id = _get_hit_field(chunk, "asset_id")
        citations.append(
            {
                "asset_id": asset_id,
                "filename": _get_hit_field(chunk, "source") or source_lookup.get(asset_id) or asset_id,
                "page": _get_hit_field(chunk, "page_num"),
                "chunk_id": cid,
                "quote": (_get_hit_field(chunk, "text") or "")[:240],
                "image_path": _get_hit_field(chunk, "image_path"),
            }
        )
    debug = RetrievalDebug(
        collection_name=store.collection,
        selected_subject_id=subject_id,
        filter_used={"subject_id": subject_id} if subject_id else None,
        top_k=top_k,
        min_score=min_score,
        query_embedding_dim=len(query_emb),
        query_embedding_min=emb_stats["min"],
        query_embedding_max=emb_stats["max"],
        query_embedding_mean=emb_stats["mean"],
        query_embedding_has_nan=emb_stats["has_nan"],
        hit_count_raw=len(hits),
        hit_count_after_filter=len(filtered_hits),
        top_hits_preview=[
            {
                "score": _get_hit_field(h, "score"),
                "page_num": _get_hit_field(h, "page_num"),
                "chunk_id": _get_hit_field(h, "chunk_id"),
                "preview": (_get_hit_field(h, "text") or _get_hit_field(h, "preview") or "")[:80],
            }
            for h in filtered_hits[:5]
        ],
        filter_retried_without_subject=filter_retried,
    ).to_dict()

    return {
        "answer": answer_text,
        "citations": citations,
        "context_expanded": extra_neighbors,
        "context_chunks": len(context_chunks),
        "debug": debug,
    }


__all__ = ["ask", "_validate_embedding"]
