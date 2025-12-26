"""Services for generating, storing, and updating per-asset notes."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import List, Optional, Tuple

from jinja2 import Template

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute
from rag_assistant.llm.provider import generate_answer
from rag_assistant.rag import judge
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore
from rag_assistant.services import asset_service
from rag_assistant.services.notes_quality import run_quality_loop
from rag_assistant.vectorstore.point_id import make_point_uuid
from rag_assistant.web import search_client


logger = logging.getLogger("rag_assistant.notes")


def _log(cfg, message: str) -> None:
    notes_cfg = getattr(cfg, "notes", None)
    if notes_cfg and getattr(notes_cfg, "debug", False):
        logger.info(message)


def _trace(trace: Optional[list], message: str) -> None:
    if trace is not None:
        trace.append(message)


def _notes_generation_params(cfg) -> dict:
    notes_cfg = getattr(cfg, "notes", None)
    gen_cfg = getattr(notes_cfg, "generation", None) if notes_cfg else None
    return {
        "temperature": getattr(gen_cfg, "temperature", None),
        "top_p": getattr(gen_cfg, "top_p", None),
        "seed": getattr(gen_cfg, "seed", None),
        "max_tokens": getattr(gen_cfg, "max_tokens", None),
        "target_chars": getattr(gen_cfg, "target_chars", None),
        "min_chars": getattr(gen_cfg, "min_chars", None),
    }


def _load_notes_prompt() -> Template:
    prompt = """
You are a study assistant. Create detailed, study notes in markdown format from the provided slides text{% if web_context %} and the external web snippets{% endif %}.

Slides content:
{{slides_context}}

{% if web_context %}External references (web):
{{web_context}}
{% endif %}

Instructions:
- Aim for about {{target_chars}} characters (minimum {{min_chars}}); make the notes comprehensive, detailed, easy to understand yet academically professional. 
- Use clear Markdown headings and bullets.
- Include definitions, intuition, detailed stepwise derivations, worked examples, pitfalls, and recap questions where relevant and grounded.
- Include key formulas only when present in the slides.
- Encase the formulas and code snippets in LaTeX math delimiters ($...$ for inline, $$...$$ for block, ''' for code) for proper rendering.
- Add concise explanations and relationships between ideas.
- Include an "Exam Tips" section grounded in the content.
- Stay strictly grounded in provided content. Do not invent facts.
- If web snippets are provided, integrate them into the notes at the appropriate places that cite snippets inline using [web:url].

Return only Markdown.
"""
    return Template(prompt)


def _load_notes_critique_prompt() -> Template:
    prompt = """
You are reviewing draft study notes. Improve clarity, structure, and completeness at all the places needed.

Draft notes:
{{draft}}

Instructions:
- Keep Markdown headings/bullets concise.
- Add missing key points if needed, based on the draft context (do not invent new topics).
- Ensure sections are organized, easy to understand and readable while being academically professional.
- Fix formatting issues; keep code/terms monospaced if present and ensure all the mathematical formulas and derivations/code snippets are properly formatted and enclosed in appropriate LaTeX math delimiters ($...$ for inline, $$...$$ for block, ''' for code).
Return the revised Markdown only.
"""
    return Template(prompt)


def _load_chunks_for_asset(subject_id: str, asset_id: str) -> List[dict]:
    sql = """
    SELECT chunk_id, subject_id, asset_id, page_num, text, start_block
    FROM chunks
    WHERE subject_id = ? AND asset_id = ?
    ORDER BY page_num ASC, start_block ASC;
    """
    return execute(asset_service.get_db_path(), sql, (subject_id, asset_id), fetchall=True) or []


def _build_slides_context(chunks: List[dict], max_chars: int = 8000) -> str:
    parts = []
    for ch in chunks:
        page = ch.get("page_num")
        prefix = f"[page {page}] " if page is not None else ""
        parts.append(f"{prefix}{ch.get('text', '')}")
    context = "\n\n".join(parts)
    return context[:max_chars]


def _chunk_markdown(markdown: str, notes_id: str, max_chars: int) -> List[dict]:
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Overview"
    buffer: List[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("#"):
            if buffer:
                sections.append((current_title, buffer))
                buffer = []
            current_title = line.lstrip("#").strip() or "Section"
            continue
        buffer.append(line)
        if sum(len(l) for l in buffer) >= max_chars:
            sections.append((current_title, buffer))
            buffer = []
    if buffer:
        sections.append((current_title, buffer))

    chunks: List[dict] = []
    for idx, (title, lines) in enumerate(sections):
        text = "\n".join(lines).strip()
        if not text:
            continue
        # ensure very long sections are split
        start = 0
        while start < len(text):
            part = text[start : start + max_chars]
            chunk_identity = f"{notes_id}:{title}:{idx}:{start}"
            chunk_hash = hashlib.sha256(chunk_identity.encode("utf-8")).hexdigest()[:20]
            chunks.append(
                {
                    "notes_chunk_id": chunk_hash,
                    "section_title": title,
                    "text": part.strip(),
                }
            )
            start += max_chars
    return chunks


def _store_notes(notes_id: str, subject_id: str, asset_id: str, markdown: str, version: int, generated_by: str, meta: dict | None) -> None:
    now = time.time()
    execute(
        asset_service.get_db_path(),
        """
        INSERT OR REPLACE INTO notes (notes_id, subject_id, asset_id, version, markdown, generated_by, created_at, updated_at, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM notes WHERE notes_id = ?), ?), ?, ?);
        """,
        (
            notes_id,
            subject_id,
            asset_id,
            version,
            markdown,
            generated_by,
            notes_id,
            now,
            now,
            json.dumps(meta or {}),
        ),
    )


def _rebuild_chunks(
    notes_id: str,
    subject_id: str,
    asset_id: str,
    markdown: str,
    chunk_char_limit: int,
    version: int,
    *,
    chunk_labels: List[str] | None = None,
    section_citations: dict | None = None,
) -> List[dict]:
    execute(asset_service.get_db_path(), "DELETE FROM notes_chunks WHERE notes_id = ?;", (notes_id,))
    chunks = _chunk_markdown(markdown, notes_id, chunk_char_limit)
    now = time.time()
    labels_iter = iter(chunk_labels or [])
    section_citations = section_citations or {}
    for ch in chunks:
        ch["version"] = version
        try:
            ch["source_label"] = next(labels_iter)
        except StopIteration:
            pass
        if ch.get("section_title"):
            ch["web_urls"] = section_citations.get(ch["section_title"])
        execute(
            asset_service.get_db_path(),
            """
            INSERT OR REPLACE INTO notes_chunks (notes_chunk_id, notes_id, subject_id, asset_id, section_title, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (ch["notes_chunk_id"], notes_id, subject_id, asset_id, ch.get("section_title"), ch["text"], now),
        )
    return chunks


def _embed_and_upsert_notes(
    chunks: List[dict],
    subject_id: str,
    asset_id: str,
    notes_id: str,
    generated_by: str,
    version: int,
    cfg,
    trace: Optional[list] = None,
) -> int:
    _log(cfg, f"[NOTES] embedding chunks n={len(chunks)}")
    _trace(trace, f"[NOTES] index:embed:start chunks={len(chunks)}")
    embedder = Embedder(config=cfg)
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    dim = len(vectors[0]) if vectors else 0
    _log(cfg, f"[NOTES] embedding complete dim={dim}")
    _trace(trace, f"[NOTES] index:embed:done dim={dim}")
    store = QdrantStore()
    store.delete_by_notes_id(notes_id)
    payloads = []
    ids = []
    for chunk, vec in zip(chunks, vectors):
        label = chunk.get("source_label") or ("From User Notes" if generated_by == "user" else "Generated Notes")
        payloads.append(
            {
                "source_type": "notes",
                "source": asset_id,
                "source_label": label,
                "subject_id": subject_id,
                "asset_id": asset_id,
                "notes_id": notes_id,
                "version": version,
                "notes_chunk_id": chunk["notes_chunk_id"],
                "chunk_id": chunk["notes_chunk_id"],
                "section_title": chunk.get("section_title"),
                "text": chunk["text"],
                "preview": chunk["text"][:240],
                "web_urls": chunk.get("web_urls") or [],
            }
        )
        ids.append(make_point_uuid(f"notes:{chunk['notes_chunk_id']}"))
    store.upsert_chunks(vectors, payloads, ids)
    _log(cfg, f"[NOTES] qdrant upsert notes_id={notes_id} version={version} points={len(ids)}")
    _trace(trace, f"[NOTES] index:qdrant_upsert chunks={len(ids)} notes_id={notes_id} version={version}")
    return len(chunks)


def _maybe_search_web(question: str, rag_hits: List[dict], cfg, trace: Optional[list] = None):
    web_cfg = getattr(cfg, "web", None)
    if not web_cfg or not getattr(web_cfg, "enabled", False):
        return {"context": "", "citations": [], "used_web": False, "queries_attempted": 0, "error": None}

    debug = {
        "hit_count_after_filter": len(rag_hits),
        "top_hits_preview": [{"score": 1.0}] if rag_hits else [],
    }
    decision = judge.should_search_web(
        question,
        rag_hits,
        debug,
        config=cfg,
        force_even_if_rag_strong=getattr(web_cfg, "force_even_if_rag_strong", False),
    )
    _trace(trace, f"[NOTES] judge:start")
    if not decision.do_search:
        _trace(trace, f"[NOTES] judge:done decision=no_search")
        return {"context": "", "citations": [], "used_web": False, "queries_attempted": 0, "error": None}

    queries = (decision.suggested_queries or [question])[: getattr(web_cfg, "max_web_queries_per_question", 2)]
    web_results_all = []
    error = None
    for q in queries:
        try:
            web_results_all.extend(
                search_client.search(
                    q,
                    config=cfg,
                    allowlist=getattr(web_cfg, "allowed_domains", []) if web_cfg else [],
                    blocklist=getattr(web_cfg, "blocked_domains", []) if web_cfg else [],
                )
            )
        except Exception as exc:
            error = str(exc)
            break
    _log(cfg, f"[NOTES] web search attempted queries={len(queries)} results={len(web_results_all)}")

    seen_urls = set()
    citations = []
    snippets = []
    for res in web_results_all:
        if res.url in seen_urls:
            continue
        seen_urls.add(res.url)
        snippets.append(f"[web:{res.url}] {res.title} — {res.snippet}")
        citations.append(
            {
                "type": "web",
                "title": res.title,
                "url": res.url,
                "quote": res.snippet,
                "snippet": res.snippet,
                "source": res.source,
            }
        )
    context = "\n".join(snippets)[:1200]
    _trace(trace, f"[NOTES] judge:done decision=search queries={len(queries)} results={len(citations)}")
    return {
        "context": context,
        "citations": citations,
        "used_web": bool(citations),
        "queries_attempted": len(queries),
        "error": error,
    }


def generate_notes_for_asset(subject_id: str, asset_id: str, config=None, trace: Optional[list] = None) -> dict:
    cfg = config or load_config()
    if trace is None:
        trace = getattr(cfg, "_notes_trace", None)
    existing = get_latest_notes(subject_id, asset_id, cfg)
    action = "regenerate" if existing else "generate"
    _log(cfg, f"[NOTES] start {action}_notes_for_asset asset_id={asset_id} subject_id={subject_id}")
    _trace(trace, f"[NOTES] start {action}_notes_for_asset asset_id={asset_id} subject_id={subject_id}")
    asset = asset_service.get_asset(asset_id)
    if not asset or asset.get("subject_id") != subject_id:
        raise ValueError(f"Asset '{asset_id}' not found for subject '{subject_id}'")

    chunks = _load_chunks_for_asset(subject_id, asset_id)
    if not chunks:
        raise ValueError(f"No indexed chunks found for asset '{asset_id}'. Run indexing first.")
    _log(cfg, f"[NOTES] fetch slide context chunks={len(chunks)}")
    _trace(trace, f"[NOTES] fetch slide context chunks={len(chunks)}")

    slides_context = _build_slides_context(chunks)
    web = _maybe_search_web(
        f"Generate study notes for asset {asset.get('original_filename') or asset_id}", chunks, cfg, trace=trace
    )
    gen_params = _notes_generation_params(cfg)
    prompt = _load_notes_prompt().render(
        slides_context=slides_context,
        web_context=web["context"],
        target_chars=gen_params.get("target_chars") or 8000,
        min_chars=gen_params.get("min_chars") or 6000,
    )
    if gen_params.get("seed") is not None and getattr(cfg.llm, "provider", "").lower() != "ollama":
        _trace(trace, f"[NOTES] warn seed_not_supported provider={getattr(cfg.llm, 'provider', 'unknown')}")
    _log(cfg, "[NOTES] LLM generating initial notes")
    _trace(trace, "[NOTES] draft_generate:start")
    draft_md = generate_answer(prompt, cfg, **gen_params)
    _trace(trace, f"[NOTES] draft_generate:done chars={len(draft_md)}")
    _log(cfg, "[NOTES] reviser improving notes")
    markdown, quality_meta = run_quality_loop(
        draft_md,
        cfg,
        trace=trace,
        base_query=asset.get("original_filename") or asset_id,
        slide_context=slides_context,
    )

    notes_id = existing["notes_id"] if existing else uuid.uuid4().hex
    version = int(existing["version"]) + 1 if existing else 1
    meta = {
        "used_web": web["used_web"] or quality_meta.get("used_web", False),
        "web_citations": web["citations"],
        "queries_attempted": web["queries_attempted"] + quality_meta.get("web_queries", 0),
        "web_error": web["error"] or quality_meta.get("web_error"),
        "quality_web_results": quality_meta.get("web_results", 0),
        "section_citations": quality_meta.get("section_citations", {}),
        "section_queries": quality_meta.get("section_queries", {}),
    }
    _store_notes(notes_id, subject_id, asset_id, markdown, version=version, generated_by="llm", meta=meta)
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    default_label = "Generated Notes"
    note_chunks = _rebuild_chunks(
        notes_id,
        subject_id,
        asset_id,
        markdown,
        chunk_limit,
        version,
        chunk_labels=[default_label] * len(_chunk_markdown(markdown, notes_id, chunk_limit)),
        section_citations=meta.get("section_citations") or {},
    )
    _log(cfg, f"[NOTES] chunking notes chunks={len(note_chunks)}")
    _trace(trace, f"[NOTES] chunking notes chunks={len(note_chunks)}")
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by="llm", version=version, cfg=cfg, trace=trace
    )
    meta["chunk_labels"] = [{"text": ch["text"], "label": ch.get("source_label", default_label)} for ch in note_chunks]
    _trace(trace, f"[NOTES] persist:notes_saved notes_id={notes_id} version={version}")
    _trace(
        trace,
        f"[NOTES] done generate used_web={meta['used_web']} web_queries={meta['queries_attempted']} web_results={meta.get('quality_web_results', len(web['citations']))}",
    )
    _log(
        cfg,
        f"[NOTES] done generate used_web={meta['used_web']} web_queries={meta['queries_attempted']} web_results={meta.get('quality_web_results', len(web['citations']))}",
    )

    return {
        "notes_id": notes_id,
        "version": version,
        "used_web": meta["used_web"],
        "web_citations_count": len(web["citations"]) + meta.get("quality_web_results", 0),
        "chunk_count": chunk_count,
    }


def get_latest_notes(subject_id: str, asset_id: str, config=None) -> dict | None:
    _ = config or load_config()
    sql = """
    SELECT * FROM notes
    WHERE subject_id = ? AND asset_id = ?
    ORDER BY version DESC
    LIMIT 1;
    """
    return execute(asset_service.get_db_path(), sql, (subject_id, asset_id), fetchone=True)


def update_notes(notes_id: str, new_markdown: str, edited_by: str = "user", config=None) -> dict:
    cfg = config or load_config()
    _log(cfg, f"[NOTES] start update_notes notes_id={notes_id}")
    trace: list[str] | None = getattr(cfg, "_notes_trace", None)
    _trace(trace, f"[NOTES] start update_notes notes_id={notes_id}")
    current = execute(asset_service.get_db_path(), "SELECT * FROM notes WHERE notes_id = ?;", (notes_id,), fetchone=True)
    if not current:
        raise ValueError(f"Notes '{notes_id}' not found")

    version = int(current.get("version", 1)) + 1
    subject_id = current["subject_id"]
    asset_id = current["asset_id"]
    generated_by = "user" if edited_by == "user" else "llm"
    meta_json = current.get("meta_json")
    meta = json.loads(meta_json) if meta_json else {}

    prev_rows = execute(asset_service.get_db_path(), "SELECT text FROM notes_chunks WHERE notes_id = ?;", (notes_id,), fetchall=True) or []
    prev_chunks_texts = [r.get("text", "") for r in prev_rows]
    prev_labels = meta.get("chunk_labels") or []

    def _normalize(text: str) -> str:
        return " ".join((text or "").split())

    label_pool: dict[str, list[str]] = {}
    if prev_labels:
        for entry in prev_labels:
            norm = _normalize(entry.get("text", ""))
            label_pool.setdefault(norm, []).append(entry.get("label") or "Generated Notes")
    else:
        default_prev_label = "From User Notes" if current.get("generated_by") == "user" else "Generated Notes"
        for txt in prev_chunks_texts:
            norm = _normalize(txt)
            label_pool.setdefault(norm, []).append(default_prev_label)

    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    # generate chunks first to know length for labels
    temp_chunks = _chunk_markdown(new_markdown, notes_id, chunk_limit)
    resolved_labels: list[str] = []
    for ch in temp_chunks:
        norm = _normalize(ch["text"])
        labels = label_pool.get(norm) or []
        if labels:
            resolved_labels.append(labels.pop(0))
        else:
            resolved_labels.append("From User Notes")
    meta["chunk_labels"] = [{"text": ch["text"], "label": lbl} for ch, lbl in zip(temp_chunks, resolved_labels)]

    _store_notes(notes_id, subject_id, asset_id, new_markdown, version=version, generated_by=generated_by, meta=meta)
    _trace(trace, f"[NOTES] persist:notes_saved notes_id={notes_id} version={version}")

    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(
        notes_id,
        subject_id,
        asset_id,
        new_markdown,
        chunk_limit,
        version,
        chunk_labels=resolved_labels,
        section_citations=meta.get("section_citations") or {},
    )
    _log(cfg, f"[NOTES] chunking notes chunks={len(note_chunks)}")
    _trace(trace, f"[NOTES] chunking notes chunks={len(note_chunks)}")
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by=generated_by, version=version, cfg=cfg, trace=trace
    )
    return {"notes_id": notes_id, "version": version, "chunk_count": chunk_count}


def save_user_notes(subject_id: str, asset_id: str, markdown: str, config=None) -> dict:
    """Create or update notes authored by a user and reindex vectors."""
    cfg = config or load_config()
    trace: list[str] | None = getattr(cfg, "_notes_trace", None)
    _log(cfg, f"[NOTES] save_user_notes asset_id={asset_id} subject_id={subject_id}")
    asset = asset_service.get_asset(asset_id)
    if not asset or asset.get("subject_id") != subject_id:
        raise ValueError(f"Asset '{asset_id}' not found for subject '{subject_id}'")
    existing = get_latest_notes(subject_id, asset_id, cfg)
    if existing:
        return update_notes(existing["notes_id"], markdown, edited_by="user", config=cfg)
    notes_id = uuid.uuid4().hex
    version = 1
    meta = {"chunk_labels": []}
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    temp_chunks = _chunk_markdown(markdown, notes_id, chunk_limit)
    labels = ["From User Notes"] * len(temp_chunks)
    meta["chunk_labels"] = [{"text": ch["text"], "label": lbl} for ch, lbl in zip(temp_chunks, labels)]
    _store_notes(notes_id, subject_id, asset_id, markdown, version=version, generated_by="user", meta=meta)
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(
        notes_id, subject_id, asset_id, markdown, chunk_limit, version, chunk_labels=labels, section_citations={}
    )
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by="user", version=version, cfg=cfg, trace=trace
    )
    return {"notes_id": notes_id, "version": version, "chunk_count": chunk_count}


def reindex_notes(notes_id: str, config=None) -> dict:
    """Rebuild chunks and vectors for the latest version of a notes_id."""
    cfg = config or load_config()
    trace: list[str] | None = getattr(cfg, "_notes_trace", None)
    row = execute(asset_service.get_db_path(), "SELECT * FROM notes WHERE notes_id = ?;", (notes_id,), fetchone=True)
    if not row:
        raise ValueError(f"Notes '{notes_id}' not found")
    version = int(row.get("version", 1))
    markdown = row.get("markdown", "")
    subject_id = row["subject_id"]
    asset_id = row["asset_id"]
    meta_json = row.get("meta_json")
    meta = json.loads(meta_json) if meta_json else {}
    labels = [entry.get("label", "Generated Notes") for entry in meta.get("chunk_labels", [])]
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(
        notes_id,
        subject_id,
        asset_id,
        markdown,
        chunk_limit,
        version,
        chunk_labels=labels,
        section_citations=meta.get("section_citations") or {},
    )
    chunk_count = _embed_and_upsert_notes(
        note_chunks,
        subject_id,
        asset_id,
        notes_id,
        generated_by=row.get("generated_by", "user"),
        version=version,
        cfg=cfg,
        trace=trace,
    )
    return {"notes_id": notes_id, "version": version, "chunk_count": chunk_count}


# Friendly aliases
generate_notes = generate_notes_for_asset

__all__ = [
    "generate_notes_for_asset",
    "generate_notes",
    "get_latest_notes",
    "update_notes",
    "save_user_notes",
    "reindex_notes",
]
