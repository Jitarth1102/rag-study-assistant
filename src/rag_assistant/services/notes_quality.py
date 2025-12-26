"""Shared quality loop for notes generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from jinja2 import Template

from rag_assistant.llm.provider import generate_answer
from rag_assistant.web import search_client
from rag_assistant.web.query_builder import build_queries_for_gap, build_section_queries, build_web_queries

logger = logging.getLogger("rag_assistant.notes")
GLOBAL_ANCHORS = [
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


@dataclass
class SectionGap:
    section_title: str
    section_anchor: Optional[str]
    gap_type: str
    what_to_add: str
    priority: int = 1
    suggested_queries: Optional[List[str]] = None
    missing_topics: Optional[List[str]] = None


def _log(cfg, message: str) -> None:
    notes_cfg = getattr(cfg, "notes", None)
    if notes_cfg and getattr(notes_cfg, "debug", False):
        logger.info(message)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "-", text.strip().lower()).strip("-")
    return cleaned or "section"


def _extract_keywords(text: str, limit: int = 12) -> List[str]:
    tokens = re.findall(r"[A-Za-z][\w'-]+", text.lower())
    stop = {
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
    filtered = [t for t in tokens if t not in stop and len(t) > 2]
    freq = {}
    for tok in filtered:
        freq[tok] = freq.get(tok, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in sorted_terms[:limit]]


def split_markdown_sections(md: str) -> List[dict]:
    sections: List[dict] = []
    lines = md.splitlines()
    current_title = "Preamble"
    current_level = 1
    buffer = []
    has_heading = False
    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if heading:
            if buffer:
                sections.append(
                    {
                        "title": current_title,
                        "level": current_level,
                        "text": "\n".join(buffer).strip(),
                        "id": f"{_slugify(current_title)}-{len(sections)}",
                        "has_heading": has_heading,
                    }
                )
                buffer = []
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip()
            has_heading = True
            continue
        buffer.append(line)
    if buffer or has_heading:
        sections.append(
            {
                "title": current_title,
                "level": current_level,
                "text": "\n".join(buffer).strip(),
                "id": f"{_slugify(current_title)}-{len(sections)}",
                "has_heading": has_heading,
            }
        )
    return sections


def apply_section_patches(md: str, patches: dict) -> str:
    sections = split_markdown_sections(md)
    rebuilt = []
    for sec in sections:
        body = patches.get(sec["id"], sec["text"])
        if not body:
            continue
        # Avoid duplicating headings if the patch already contains one
        if sec.get("has_heading", False) and not re.match(r"^#{1,6}\s", body.strip()):
            rebuilt.append(f"{'#' * sec['level']} {sec['title']}\n{body}".strip())
        else:
            rebuilt.append(body.strip())
    return "\n\n".join(rebuilt)


def _critique_prompt() -> Template:
    return Template(
        """
You are reviewing draft study notes. Improve clarity, structure, and completeness.

Draft notes:
{{draft}}

Instructions:
- Keep Markdown headings/bullets concise.
- Add missing key points if needed, based on the draft context (do not invent new topics).
- Ensure sections are organized and readable.
- Add helpful examples, brief derivations, and exam-style cues where content allows; avoid fluff or repetition.
- Fix formatting issues; keep code/terms monospaced if present.
Return the revised Markdown only.
"""
    )


def _judge_prompt() -> Template:
    return Template(
        """
You are judging draft study notes for quality, completeness, and structure.

Draft notes:
{{draft}}

Instructions:
- List key missing points or weak spots in bullet form.
- If the draft is already clear and complete, say "No major issues".
- Keep feedback concise.
"""
    )


def judge_notes(draft_md: str, config, trace: Optional[list] = None, round_num: int = 1) -> dict:
    _trace = trace.append if trace is not None else lambda *a, **k: None
    _trace(f"[NOTES] judge_review:start round={round_num}")
    cfg = config
    gen_cfg = getattr(getattr(cfg, "notes", None), "generation", None) if cfg else None
    params = {
        "temperature": getattr(gen_cfg, "temperature", None),
        "top_p": getattr(gen_cfg, "top_p", None),
        "seed": getattr(gen_cfg, "seed", None),
        "max_tokens": getattr(gen_cfg, "max_tokens", None),
    }
    llm_cfg = getattr(cfg, "llm", None) if cfg else None
    provider = getattr(llm_cfg, "provider", "") if llm_cfg is not None else ""
    if params.get("seed") is not None and provider.lower() != "ollama":
        _trace("[NOTES] warn seed_not_supported provider={}".format(provider or "unknown"))
    critique = generate_answer(_judge_prompt().render(draft=draft_md), cfg, **params)
    critique_text = critique or ""
    needs_revision = not ("no major issues" in critique_text.lower())
    needs_web = "external" in critique_text.lower() or "web" in critique_text.lower()
    suggested_queries: list[str] = []
    _trace(
        f"[NOTES] judge_review:done round={round_num} items={'na' if not critique_text else len(critique_text.splitlines())} needs_revision={needs_revision}"
    )
    return {"needs_revision": needs_revision, "critique": critique_text, "needs_web": needs_web, "suggested_queries": suggested_queries}


def detect_section_gaps(markdown: str, slide_context: str, cfg, trace=None) -> List[SectionGap]:
    _trace = trace.append if trace is not None else lambda *a, **k: None
    _trace("[NOTES] gap_detect:start")
    sections = split_markdown_sections(markdown or "")
    slide_terms = _extract_keywords(slide_context or "", limit=14)
    gaps: List[SectionGap] = []
    for sec in sections:
        section_terms = _extract_keywords(sec.get("text", ""), limit=12)
        overlap = 0.0
        if section_terms or slide_terms:
            overlap = len(set(section_terms) & set(slide_terms)) / max(len(set(section_terms) | set(slide_terms)), 1)
        missing_topics = [t for t in slide_terms if t not in section_terms][:6]
        _trace(
            f"[NOTES] gap_detect:section title={sec.get('title')} overlap={overlap:.2f} missing={missing_topics}"
        )
        if missing_topics and overlap < 0.6:
            gaps.append(
                SectionGap(
                    section_title=sec["title"],
                    section_anchor=sec.get("id"),
                    gap_type="missing_topics",
                    what_to_add="Add detail on: " + ", ".join(missing_topics),
                    priority=1,
                    suggested_queries=missing_topics,
                    missing_topics=missing_topics,
                )
            )
    _trace(f"[NOTES] gap_detect:done found={len(gaps)}")
    return gaps


def _log_web_decision(
    trace: Optional[list],
    cfg,
    *,
    decision: str,
    reason: str,
    num_sections: Optional[int] = None,
    num_queries: Optional[int] = None,
):
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    web_cfg = getattr(cfg, "web", None)
    allow = getattr(notes_cfg, "web_allow_domains", None)
    block = getattr(notes_cfg, "web_block_domains", None)
    allow_count = len(allow) if allow else 0
    block_count = len(block) if block else 0
    _trace(
        f"[NOTES] notes_web_decision: decision={decision} reason={reason} mode={getattr(notes_cfg, 'web_mode', 'auto')} "
        f"global_web={getattr(web_cfg, 'enabled', False)} notes_web={getattr(notes_cfg, 'web_augmentation_enabled', False)} "
        f"max_queries_per_note={getattr(notes_cfg, 'max_web_queries_per_notes', None)} max_results_per_query={getattr(notes_cfg, 'max_web_results_per_query', None)} "
        f"allow_domains={allow_count} block_domains={block_count} num_sections={num_sections} num_queries={num_queries}"
    )


def _collect_web_context(
    judge_result: dict, cfg, trace, base_query: Optional[str] = None, subject: Optional[str] = None
) -> tuple[str, dict]:
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}
    if not judge_result.get("needs_web"):
        _log_web_decision(trace, cfg, decision="no_search", reason="judge_no_web", num_sections=None, num_queries=None)
        return "", meta
    if not getattr(getattr(cfg, "web", None), "enabled", False) or not getattr(notes_cfg, "web_augmentation_enabled", False):
        reason = "disabled_global" if not getattr(getattr(cfg, "web", None), "enabled", False) else "disabled_notes"
        _trace(f"[NOTES] web_decision needs_web=True skipped={reason}")
        _log_web_decision(trace, cfg, decision="no_search", reason=reason, num_sections=None, num_queries=None)
        return "", meta
    queries = build_web_queries(
        subject=subject,
        asset_title=base_query,
        intents=judge_result.get("suggested_queries") or [judge_result.get("critique", "")],
        question=None,
        max_queries=getattr(notes_cfg, "max_web_queries_per_notes", 2),
    )
    if not queries:
        _log_web_decision(trace, cfg, decision="no_search", reason="no_queries", num_sections=None, num_queries=None)
        return "", meta
    _trace(f"[NOTES] web_decision needs_web=True queries={len(queries)}")
    _log_web_decision(trace, cfg, decision="search", reason="judge_needs_web", num_sections=None, num_queries=len(queries))
    results: list = []
    for q in queries:
        try:
            _trace("[NOTES] web_search:start")
            query_results = search_client.search(
                q,
                config=cfg,
                allowlist=getattr(notes_cfg, "web_allow_domains", []) or getattr(getattr(cfg, "web", None), "allowed_domains", []),
                blocklist=getattr(notes_cfg, "web_block_domains", []) or getattr(getattr(cfg, "web", None), "blocked_domains", []),
                max_results=getattr(notes_cfg, "max_web_results_per_query", 5),
            )
            results.extend(query_results[: getattr(notes_cfg, "max_web_results_per_query", 5)])
        except Exception as exc:
            meta["web_error"] = str(exc)
            _trace(f"[NOTES] web_search:error {exc}")
            break
    if not results:
        return "", meta
    meta["used_web"] = True
    meta["web_queries"] = len(queries)
    meta["web_results"] = len(results)
    _trace(f"[NOTES] web_search:done web_queries={meta['web_queries']} web_results={meta['web_results']}")
    snippet_limit = getattr(notes_cfg, "web_snippet_char_limit", 400)
    context_limit = getattr(notes_cfg, "web_context_char_limit", 2500)
    parts = []
    total = 0
    for res in results:
        snippet = (res.snippet or "")[:snippet_limit]
        domain = res.url.split("/")[2] if res.url and "://" in res.url else res.url
        entry = f"- {res.title} ({domain}): {snippet}"
        if total + len(entry) > context_limit:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts), meta


def _decide_notes_web(cfg, gaps: List[SectionGap], critique_text: str, trace: Optional[list]) -> dict:
    """Notes-native decision maker (no RAG hit/score heuristics)."""
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    web_cfg = getattr(cfg, "web", None)
    enabled = bool(getattr(notes_cfg, "web_augmentation_enabled", False) and getattr(web_cfg, "enabled", False))
    mode = getattr(notes_cfg, "web_mode", "auto")
    critique_lower = (critique_text or "").lower()
    needs_external = any(
        kw in critique_lower
        for kw in [
            "external",
            "missing",
            "define",
            "definition",
            "unclear",
            "needs example",
            "needs citation",
            "needs reference",
            "background",
            "derive",
            "compare",
            "algorithm",
        ]
    )
    do_search = False
    reason = "disabled" if not enabled else "mode_skip"
    if enabled:
        if mode == "never":
            do_search = False
            reason = "mode_never"
        elif mode == "always":
            do_search = True
            reason = "mode_always"
        else:  # auto
            if gaps:
                do_search = True
                reason = "gap_detected"
            elif needs_external:
                do_search = True
                reason = "critique_requires_external"
            else:
                reason = "no_gaps_no_external_need"
    _log_web_decision(trace, cfg, decision="search" if do_search else "no_search", reason=reason, num_sections=len(gaps), num_queries=None)
    return {"do_search": do_search, "reason": reason, "mode": mode, "needs_external": needs_external}


def _section_web_augment(
    draft_md: str,
    cfg,
    gaps: List[SectionGap],
    trace: Optional[list],
    slide_context: str = "",
) -> Tuple[str, dict]:
    """Run section-targeted web augmentation and return updated markdown + meta."""
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    max_gap_queries = getattr(notes_cfg, "max_web_queries_per_notes", 2) if notes_cfg else 2
    sections = split_markdown_sections(draft_md)
    patches = {}
    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None, "section_citations": {}, "section_queries": {}}

    def _revise_section(sec_text: str, gap: SectionGap, context: str, section_id: str) -> str:
        _trace(f"[NOTES] patch_section:start section={gap.section_title}")
        gen_cfg_local = getattr(notes_cfg, "generation", None) if notes_cfg else None
        prompt = (
            "Rewrite this section only, keeping original style. Preserve the original heading and core bullets; append a short 'Additional context' subsection (max 6 lines) based on the web context.\n\n"
            f"Original Section:\n{sec_text}\n\nGap:\n{gap.what_to_add}\n\nExternal Context:\n{context}\n\n"
            "Return updated section Markdown with inline footnote citations like [^w1], [^w2] and footnotes at the end of the section as [^w1]: URL (Title)."
        )
        revised = generate_answer(
            prompt,
            cfg,
            **{
                "temperature": getattr(gen_cfg_local, "temperature", None) if gen_cfg_local else None,
                "top_p": getattr(gen_cfg_local, "top_p", None) if gen_cfg_local else None,
                "seed": getattr(gen_cfg_local, "seed", None) if gen_cfg_local else None,
                "max_tokens": getattr(gen_cfg_local, "max_tokens", None) if gen_cfg_local else None,
            },
        )
        _trace(f"[NOTES] patch_section:done section={gap.section_title}")
        return revised

    used_queries = 0
    for gap in sorted(gaps, key=lambda g: g.priority)[: max_gap_queries]:
        if used_queries >= max_gap_queries:
            break
        section_text = ""
        sections = split_markdown_sections(draft_md)
        sec = next((s for s in sections if s["title"] == gap.section_title), None)
        if sec:
            section_text = sec.get("text", "")
        queries = build_section_queries(
            gap.section_title, section_text, slide_context, max_queries=max_gap_queries - used_queries
        )
        meta["section_queries"][gap.section_title] = queries
        _trace(f"[NOTES] query_build gap={gap.section_title} queries={queries}")
        if not queries:
            continue
        context_parts = []
        anchors = set(GLOBAL_ANCHORS)
        anchors.update((gap.missing_topics or []))
        anchors.update((gap.section_title or "").lower().split())
        anchors = {a.lower() for a in anchors if a}
        kept_results = 0
        gap_queries_used = 0
        for q in queries:
            if used_queries >= max_gap_queries:
                break
            _trace(f"[NOTES] web:search section=\"{gap.section_title}\" query=\"{q[:80]}\"")
            try:
                results = search_client.search(
                    q,
                    config=cfg,
                    allowlist=getattr(notes_cfg, "web_allow_domains", []) or [],
                    blocklist=getattr(notes_cfg, "web_block_domains", []) or [],
                    max_results=getattr(notes_cfg, "max_web_results_per_query", 5),
                )
            except Exception as exc:  # pragma: no cover - best effort
                meta["web_error"] = str(exc)
                _trace(f"[NOTES] gap_web_search:error {exc}")
                continue
            used_queries += 1
            gap_queries_used += 1
            filtered_results = []
            for res in results:
                content = f"{res.title} {res.snippet}".lower()
                anchor_hits = sum(1 for a in anchors if a and a in content)
                if anchor_hits < 2 or "definitions in writing" in content:
                    _trace(f"[NOTES] patch_section:filtered_offtopic url={res.url}")
                    continue
                filtered_results.append(res)
            kept_results += len(filtered_results)
            meta["section_citations"][gap.section_title] = [r.url for r in filtered_results if getattr(r, "url", None)]
            _trace(f"[NOTES] web:done section=\"{gap.section_title}\" results={len(filtered_results)} provider={getattr(cfg.web, 'provider', '')}")
            for idx, res in enumerate(filtered_results, start=1):
                snippet = (res.snippet or "")[: getattr(notes_cfg, "web_snippet_char_limit", 400)]
                foot_id = f"w_{_slugify(gap.section_title)}_{idx}"
                footnote = f"[^{foot_id}]: {res.url} ({res.title})"
                context_parts.append(f"- {res.title} ({res.url}): {snippet}\n{footnote}")
        if context_parts and sec:
            new_text = _revise_section(sec.get("text", ""), gap, "\n".join(context_parts), sec["id"])
            banned = ["definitions in writing", "role of definitions"]
            if any(b in (new_text or "").lower() for b in banned):
                _trace(f"[NOTES] patch_section:filtered_offtopic url=banned_content")
                continue
            anchor_hits_new = sum(1 for a in anchors if a and a in (new_text or "").lower())
            if anchor_hits_new < 2:
                _trace(f"[NOTES] patch_section:filtered_offtopic url=anchor_check")
                continue
            foots = {line for p in context_parts for line in p.splitlines() if line.startswith("[^")}
            if foots:
                new_text = new_text + "\n\n" + "\n".join(foots)
            patches[sec["id"]] = new_text
            if kept_results > 0:
                meta["used_web"] = True
                meta["web_queries"] += gap_queries_used
                meta["web_results"] += kept_results
                _trace(f"[NOTES] citations:inserted count={len(context_parts)} section={gap.section_title}")
                _trace(f"[NOTES] patch_section:applied section={gap.section_title}")
    if patches:
        draft_md = apply_section_patches(draft_md, patches)
    return draft_md, meta


def _rescue_web_context(cfg, trace, critique: str, base_query: Optional[str], subject: Optional[str]) -> tuple[str, dict]:
    """Run a bounded rescue web search when still needs revision after round 2."""
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}
    if not getattr(getattr(cfg, "web", None), "enabled", False) or not getattr(notes_cfg, "web_augmentation_enabled", False):
        _log_web_decision(trace, cfg, decision="no_search", reason="rescue_disabled", num_sections=None, num_queries=None)
        return "", meta
    queries = build_web_queries(
        subject=subject,
        asset_title=base_query,
        intents=[critique] if critique else [],
        question=None,
        max_queries=getattr(notes_cfg, "max_web_queries_per_notes", 2),
    )
    if not queries:
        queries = ["study notes context"][: getattr(notes_cfg, "max_web_queries_per_notes", 2)]
    _trace(f"[NOTES] web_rescue:start reason=needs_revision_after_round2 queries={len(queries)}")
    _log_web_decision(trace, cfg, decision="search", reason="rescue", num_sections=None, num_queries=len(queries))
    results = []
    for q in queries:
        try:
            query_results = search_client.search(
                q,
                config=cfg,
                allowlist=getattr(notes_cfg, "web_allow_domains", []) or getattr(getattr(cfg, "web", None), "allowed_domains", []),
                blocklist=getattr(notes_cfg, "web_block_domains", []) or getattr(getattr(cfg, "web", None), "blocked_domains", []),
                max_results=getattr(notes_cfg, "max_web_results_per_query", 5),
            )
            results.extend(query_results[: getattr(notes_cfg, "max_web_results_per_query", 5)])
        except Exception as exc:  # pragma: no cover - rescue best effort
            meta["web_error"] = str(exc)
            _trace(f"[NOTES] web_rescue:error {exc}")
            break
    if not results:
        return "", meta
    meta["used_web"] = True
    meta["web_queries"] = len(queries)
    meta["web_results"] = len(results)
    snippet_limit = getattr(notes_cfg, "web_snippet_char_limit", 400)
    context_limit = getattr(notes_cfg, "web_context_char_limit", 2500)
    parts = []
    total = 0
    for res in results:
        snippet = (res.snippet or "")[:snippet_limit]
        domain = res.url.split("/")[2] if res.url and "://" in res.url else res.url
        entry = f"- {res.title} ({domain}): {snippet}"
        if total + len(entry) > context_limit:
            break
        parts.append(entry)
        total += len(entry)
    _trace(f"[NOTES] web_rescue:search:done results={meta['web_results']}")
    return "\n".join(parts), meta


def run_quality_loop(draft_md: str, config, trace=None, base_query: Optional[str] = None, slide_context: str = "") -> tuple[str, dict]:
    """Run a single critique + revision pass."""
    _trace = trace.append if trace is not None else lambda *a, **k: None
    cfg = config
    gen_cfg = getattr(getattr(cfg, "notes", None), "generation", None) if cfg else None
    params = {
        "temperature": getattr(gen_cfg, "temperature", None),
        "top_p": getattr(gen_cfg, "top_p", None),
        "seed": getattr(gen_cfg, "seed", None),
        "max_tokens": getattr(gen_cfg, "max_tokens", None),
    }
    min_chars = getattr(gen_cfg, "min_chars", None) or 0
    notes_cfg = getattr(cfg, "notes", None)
    max_gap_queries = getattr(notes_cfg, "max_web_queries_per_notes", 2) if notes_cfg else 2

    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}
    meta["section_citations"] = {}
    meta["section_queries"] = {}

    current = draft_md
    # Section-driven decision (notes-native)
    gaps = detect_section_gaps(current, slide_context or "", cfg, trace=trace) or []
    judge_result = judge_notes(current, cfg, trace=trace, round_num=1)
    web_decision = _decide_notes_web(cfg, gaps, judge_result.get("critique", ""), trace)
    if web_decision.get("do_search"):
        if not gaps and split_markdown_sections(current):
            # fabricate a catch-all gap so we still run targeted search
            first = split_markdown_sections(current)[0]
            gaps = [
                SectionGap(
                    section_title=first["title"],
                    section_anchor=first.get("id"),
                    gap_type="general",
                    what_to_add="Add definitions or examples to strengthen this section.",
                    priority=1,
                    suggested_queries=[judge_result.get("critique", "")],
                )
            ]
        current, section_meta = _section_web_augment(current, cfg, gaps, trace, slide_context=slide_context)
        # merge meta
        meta["used_web"] = meta["used_web"] or section_meta.get("used_web", False)
        meta["web_queries"] += section_meta.get("web_queries", 0)
        meta["web_results"] += section_meta.get("web_results", 0)
        meta["web_error"] = meta["web_error"] or section_meta.get("web_error")
        meta["section_citations"].update(section_meta.get("section_citations", {}))
        meta["section_queries"].update(section_meta.get("section_queries", {}))

    web_context = ""
    web_meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}

    def _revise(text: str, critique_text: str, round_num: int, expand_for_length: bool = False, rescue: bool = False) -> str:
        tag = (
            f"round={round_num}"
            if not expand_for_length and not rescue
            else "expand_for_length"
            if expand_for_length
            else "rescue_with_web_context"
        )
        _trace(f"[NOTES] revise:start {tag} with_web_context={bool(web_context)}")
        prompt_text = text
        if expand_for_length:
            prompt_text = (
                text
                + "\n\nThe notes are shorter than desired. Expand missing sections with examples, clarifications, exam tips; avoid fluff."
            )
        draft_for_prompt = prompt_text + ("\n\nCritique:\n" + critique_text if critique_text else "")
        if web_context:
            draft_for_prompt += "\n\nExternal Context:\n" + web_context
        critique_prompt = _critique_prompt().render(draft=draft_for_prompt)
        revised = generate_answer(critique_prompt, cfg, **params)
        _trace(f"[NOTES] revise:done {tag}")
        return revised

    if judge_result.get("needs_revision", True):
        if not meta["used_web"]:
            web_context, web_meta = _collect_web_context(judge_result, cfg, trace, base_query=base_query)
            meta["used_web"] = meta["used_web"] or web_meta.get("used_web", False)
            meta["web_queries"] += web_meta.get("web_queries", 0)
            meta["web_results"] += web_meta.get("web_results", 0)
            meta["web_error"] = meta["web_error"] or web_meta.get("web_error")
        current = _revise(current, judge_result.get("critique", ""), round_num=1)

    # Round 2 judge
    judge_result2 = judge_notes(current, cfg, trace=trace, round_num=2)
    if judge_result2.get("needs_revision", False):
        if not meta["used_web"]:
            web_context, web_meta = _collect_web_context(judge_result2, cfg, trace, base_query=base_query, subject=None)
            meta["used_web"] = meta["used_web"] or web_meta.get("used_web", False)
            meta["web_queries"] += web_meta.get("web_queries", 0)
            meta["web_results"] += web_meta.get("web_results", 0)
            meta["web_error"] = meta["web_error"] or web_meta.get("web_error")
            if web_context:
                current = _revise(current, judge_result2.get("critique", ""), round_num=2, rescue=True)
            else:
                current = _revise(current, judge_result2.get("critique", ""), round_num=2)
        else:
            current = _revise(current, judge_result2.get("critique", ""), round_num=2)
        # Final rescue if still needs revision and web unused
        if judge_result2.get("needs_revision", False) and not meta["used_web"]:
            rescue_context, rescue_meta = _rescue_web_context(
                cfg, trace, judge_result2.get("critique", ""), base_query, subject=None
            )
            meta["used_web"] = meta["used_web"] or rescue_meta.get("used_web", False)
            meta["web_queries"] += rescue_meta.get("web_queries", 0)
            meta["web_results"] += rescue_meta.get("web_results", 0)
            meta["web_error"] = meta["web_error"] or rescue_meta.get("web_error")
            if rescue_context:
                web_context = rescue_context
                current = _revise(current, judge_result2.get("critique", ""), round_num=2, rescue=True)

    # Optional expansion only if min_chars > 0 and still short
    if min_chars > 0 and len(current or "") < min_chars:
        current = _revise(current, "", round_num=2, expand_for_length=True)

    return current, meta


__all__ = ["run_quality_loop", "judge_notes"]
