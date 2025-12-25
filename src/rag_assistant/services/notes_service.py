"""Services for generating, storing, and updating per-asset notes."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import List, Tuple

from jinja2 import Template

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute
from rag_assistant.llm.provider import generate_answer
from rag_assistant.rag import judge
from rag_assistant.retrieval.embedder import Embedder
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore
from rag_assistant.services import asset_service
from rag_assistant.vectorstore.point_id import make_point_uuid
from rag_assistant.web import search_client


def _load_notes_prompt() -> Template:
    prompt = """
You are a study assistant. Create well-structured Markdown study notes from the provided slides text{% if web_context %} and the external web snippets{% endif %}.

Slides content:
{{slides_context}}

{% if web_context %}External references (web):
{{web_context}}
{% endif %}

Instructions:
- Use clear Markdown headings and bullets.
- Include definitions and key formulas only when present in the slides.
- Add concise explanations and relationships between ideas.
- Include an "Exam Tips" section grounded in the content.
- Stay strictly grounded in provided content. Do not invent facts.
- If web snippets are provided, add a section titled "## External Additions (Web)" that cites snippets inline using [web:url].

Return only Markdown.
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


def _rebuild_chunks(notes_id: str, subject_id: str, asset_id: str, markdown: str, chunk_char_limit: int, version: int) -> List[dict]:
    execute(asset_service.get_db_path(), "DELETE FROM notes_chunks WHERE notes_id = ?;", (notes_id,))
    chunks = _chunk_markdown(markdown, notes_id, chunk_char_limit)
    now = time.time()
    for ch in chunks:
        ch["version"] = version
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
) -> int:
    embedder = Embedder(config=cfg)
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    store = QdrantStore()
    store.delete_by_notes_id(notes_id)
    payloads = []
    ids = []
    label = "From User Notes" if generated_by == "user" else "Generated Notes"
    for chunk, vec in zip(chunks, vectors):
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
            }
        )
        ids.append(make_point_uuid(f"notes:{chunk['notes_chunk_id']}"))
    store.upsert_chunks(vectors, payloads, ids)
    return len(chunks)


def _maybe_search_web(question: str, rag_hits: List[dict], cfg):
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
    if not decision.do_search:
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
    return {
        "context": context,
        "citations": citations,
        "used_web": bool(citations),
        "queries_attempted": len(queries),
        "error": error,
    }


def generate_notes_for_asset(subject_id: str, asset_id: str, config=None) -> dict:
    cfg = config or load_config()
    asset = asset_service.get_asset(asset_id)
    if not asset or asset.get("subject_id") != subject_id:
        raise ValueError(f"Asset '{asset_id}' not found for subject '{subject_id}'")

    chunks = _load_chunks_for_asset(subject_id, asset_id)
    if not chunks:
        raise ValueError(f"No indexed chunks found for asset '{asset_id}'. Run indexing first.")

    slides_context = _build_slides_context(chunks)
    web = _maybe_search_web(f"Generate study notes for asset {asset.get('original_filename') or asset_id}", chunks, cfg)
    prompt = _load_notes_prompt().render(slides_context=slides_context, web_context=web["context"])
    markdown = generate_answer(prompt, cfg)

    existing = get_latest_notes(subject_id, asset_id, cfg)
    notes_id = existing["notes_id"] if existing else uuid.uuid4().hex
    version = int(existing["version"]) + 1 if existing else 1
    meta = {
        "used_web": web["used_web"],
        "web_citations": web["citations"],
        "queries_attempted": web["queries_attempted"],
        "web_error": web["error"],
    }
    _store_notes(notes_id, subject_id, asset_id, markdown, version=version, generated_by="llm", meta=meta)
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(notes_id, subject_id, asset_id, markdown, chunk_limit, version)
    chunk_count = _embed_and_upsert_notes(note_chunks, subject_id, asset_id, notes_id, generated_by="llm", version=version, cfg=cfg)

    return {
        "notes_id": notes_id,
        "version": version,
        "used_web": web["used_web"],
        "web_citations_count": len(web["citations"]),
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
    current = execute(asset_service.get_db_path(), "SELECT * FROM notes WHERE notes_id = ?;", (notes_id,), fetchone=True)
    if not current:
        raise ValueError(f"Notes '{notes_id}' not found")

    version = int(current.get("version", 1)) + 1
    subject_id = current["subject_id"]
    asset_id = current["asset_id"]
    generated_by = "user" if edited_by == "user" else "llm"
    meta_json = current.get("meta_json")
    meta = json.loads(meta_json) if meta_json else {}
    _store_notes(notes_id, subject_id, asset_id, new_markdown, version=version, generated_by=generated_by, meta=meta)

    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(notes_id, subject_id, asset_id, new_markdown, chunk_limit, version)
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by=generated_by, version=version, cfg=cfg
    )
    return {"notes_id": notes_id, "version": version, "chunk_count": chunk_count}


def save_user_notes(subject_id: str, asset_id: str, markdown: str, config=None) -> dict:
    """Create or update notes authored by a user and reindex vectors."""
    cfg = config or load_config()
    asset = asset_service.get_asset(asset_id)
    if not asset or asset.get("subject_id") != subject_id:
        raise ValueError(f"Asset '{asset_id}' not found for subject '{subject_id}'")
    existing = get_latest_notes(subject_id, asset_id, cfg)
    if existing:
        return update_notes(existing["notes_id"], markdown, edited_by="user", config=cfg)
    notes_id = uuid.uuid4().hex
    version = 1
    _store_notes(notes_id, subject_id, asset_id, markdown, version=version, generated_by="user", meta={})
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(notes_id, subject_id, asset_id, markdown, chunk_limit, version)
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by="user", version=version, cfg=cfg
    )
    return {"notes_id": notes_id, "version": version, "chunk_count": chunk_count}


def reindex_notes(notes_id: str, config=None) -> dict:
    """Rebuild chunks and vectors for the latest version of a notes_id."""
    cfg = config or load_config()
    row = execute(asset_service.get_db_path(), "SELECT * FROM notes WHERE notes_id = ?;", (notes_id,), fetchone=True)
    if not row:
        raise ValueError(f"Notes '{notes_id}' not found")
    version = int(row.get("version", 1))
    markdown = row.get("markdown", "")
    subject_id = row["subject_id"]
    asset_id = row["asset_id"]
    chunk_limit = min(getattr(cfg.ingest, "max_chunk_chars", 800), 1200)
    note_chunks = _rebuild_chunks(notes_id, subject_id, asset_id, markdown, chunk_limit, version)
    chunk_count = _embed_and_upsert_notes(
        note_chunks, subject_id, asset_id, notes_id, generated_by=row.get("generated_by", "user"), version=version, cfg=cfg
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
