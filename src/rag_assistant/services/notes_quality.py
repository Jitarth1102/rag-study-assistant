"""Shared quality loop for notes generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from jinja2 import Template

from rag_assistant.llm.provider import generate_answer
from rag_assistant.web import search_client
from rag_assistant.web.query_builder import build_queries_for_gap, build_web_queries

logger = logging.getLogger("rag_assistant.notes")


@dataclass
class SectionGap:
    section_title: str
    section_anchor: Optional[str]
    gap_type: str
    what_to_add: str
    priority: int = 1
    suggested_queries: Optional[List[str]] = None


def _log(cfg, message: str) -> None:
    notes_cfg = getattr(cfg, "notes", None)
    if notes_cfg and getattr(notes_cfg, "debug", False):
        logger.info(message)


def split_markdown_sections(md: str) -> List[dict]:
    sections: List[dict] = []
    lines = md.splitlines()
    current_title = "Preamble"
    current_level = 1
    buffer = []
    idx = 0
    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if heading:
            if buffer:
                sections.append(
                    {"title": current_title, "level": current_level, "text": "\n".join(buffer).strip(), "id": f"{current_title.lower().replace(' ', '-')}-{len(sections)}"}
                )
                buffer = []
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip()
            continue
        buffer.append(line)
        idx += 1
    if buffer:
        sections.append(
            {"title": current_title, "level": current_level, "text": "\n".join(buffer).strip(), "id": f"{current_title.lower().replace(' ', '-')}-{len(sections)}"}
        )
    return sections


def apply_section_patches(md: str, patches: dict) -> str:
    sections = split_markdown_sections(md)
    rebuilt = []
    for sec in sections:
        if sec["id"] in patches:
            rebuilt.append(patches[sec["id"]])
        else:
            rebuilt.append(sec["text"])
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
    # placeholder simple heuristic: no structured gaps unless caller monkeypatches
    _trace("[NOTES] gap_detect:done found=0")
    return []


def _log_web_decision(
    trace: Optional[list],
    cfg,
    *,
    decision: str,
    reason: str,
    hit_count: Optional[int] = None,
    best_score: Optional[float] = None,
):
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    web_cfg = getattr(cfg, "web", None)
    allow = getattr(notes_cfg, "web_allow_domains", None)
    block = getattr(notes_cfg, "web_block_domains", None)
    allow_count = len(allow) if allow else 0
    block_count = len(block) if block else 0
    _trace(
        f"[NOTES] judge:done decision={decision} reason={reason} hit_count={hit_count} "
        f"best_score={best_score} global_web={getattr(web_cfg, 'enabled', False)} "
        f"notes_web={getattr(notes_cfg, 'web_augmentation_enabled', False)} "
        f"min_hits_to_skip_web={getattr(web_cfg, 'min_rag_hits_to_skip_web', None)} "
        f"min_score_to_skip_web={getattr(web_cfg, 'min_rag_score_to_skip_web', None)} "
        f"force_even_if_rag_strong={getattr(web_cfg, 'force_even_if_rag_strong', None)} "
        f"allow_domains={allow_count} block_domains={block_count}"
    )


def _collect_web_context(
    judge_result: dict, cfg, trace, base_query: Optional[str] = None, subject: Optional[str] = None
) -> tuple[str, dict]:
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}
    if not judge_result.get("needs_web"):
        _log_web_decision(trace, cfg, decision="no_search", reason="judge_no_web", hit_count=None, best_score=None)
        return "", meta
    if not getattr(getattr(cfg, "web", None), "enabled", False) or not getattr(notes_cfg, "web_augmentation_enabled", False):
        reason = "disabled_global" if not getattr(getattr(cfg, "web", None), "enabled", False) else "disabled_notes"
        _trace(f"[NOTES] web_decision needs_web=True skipped={reason}")
        _log_web_decision(trace, cfg, decision="no_search", reason=reason, hit_count=None, best_score=None)
        return "", meta
    queries = build_web_queries(
        subject=subject,
        asset_title=base_query,
        intents=judge_result.get("suggested_queries") or [judge_result.get("critique", "")],
        question=None,
        max_queries=getattr(notes_cfg, "max_web_queries_per_notes", 2),
    )
    if not queries:
        _log_web_decision(trace, cfg, decision="no_search", reason="no_queries", hit_count=None, best_score=None)
        return "", meta
    _trace(f"[NOTES] web_decision needs_web=True queries={len(queries)}")
    _log_web_decision(trace, cfg, decision="search", reason="judge_needs_web", hit_count=None, best_score=None)
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


def _rescue_web_context(cfg, trace, critique: str, base_query: Optional[str], subject: Optional[str]) -> tuple[str, dict]:
    """Run a bounded rescue web search when still needs revision after round 2."""
    _trace = trace.append if trace is not None else lambda *a, **k: None
    notes_cfg = getattr(cfg, "notes", None)
    meta = {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}
    if not getattr(getattr(cfg, "web", None), "enabled", False) or not getattr(notes_cfg, "web_augmentation_enabled", False):
        _log_web_decision(trace, cfg, decision="no_search", reason="rescue_disabled", hit_count=None, best_score=None)
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
    _log_web_decision(trace, cfg, decision="search", reason="rescue", hit_count=None, best_score=None)
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


def run_quality_loop(draft_md: str, config, trace=None, base_query: Optional[str] = None) -> tuple[str, dict]:
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

    # Section-driven augmentation before judge
    gaps = detect_section_gaps(draft_md, "", cfg, trace=trace) or []
    if gaps:
        _trace(f"[NOTES] gap_detect:found {len(gaps)}")
        sections = split_markdown_sections(draft_md)
        patches = {}
        used_queries = 0

        def _revise_section(sec_text: str, gap: SectionGap, context: str, section_id: str) -> str:
            _trace(f"[NOTES] patch_section:start section={gap.section_title}")
            prompt = (
                "Rewrite this section only, keeping original style. Address the gap described.\n\n"
                f"Original Section:\n{sec_text}\n\nGap:\n{gap.what_to_add}\n\nExternal Context:\n{context}\n\n"
                "Return updated section Markdown with inline footnote citations like [^w1], [^w2] and footnotes at the end of the section as [^w1]: URL (Title)."
            )
            revised = generate_answer(prompt, cfg, **params)
            _trace(f"[NOTES] patch_section:done section={gap.section_title}")
            return revised

        for gap in sorted(gaps, key=lambda g: g.priority)[: max_gap_queries]:
            queries = build_queries_for_gap(gap, None, max_queries=max_gap_queries)
            meta["section_queries"][gap.section_title] = queries
            _trace(f"[NOTES] query_build gap={gap.section_title} queries={queries}")
            if not queries:
                continue
            context_parts = []
            for q in queries:
                if used_queries >= max_gap_queries:
                    break
                _trace(f"[NOTES] gap_web_search:start query={q}")
                try:
                    results = search_client.search(
                        q,
                        config=cfg,
                        allowlist=getattr(notes_cfg, "web_allow_domains", []) or [],
                        blocklist=getattr(notes_cfg, "web_block_domains", []) or [],
                        max_results=getattr(notes_cfg, "max_web_results_per_query", 5),
                    )
                except Exception as exc:  # pragma: no cover
                    _trace(f"[NOTES] gap_web_search:error {exc}")
                    continue
                used_queries += 1
                meta["used_web"] = True
                meta["web_queries"] += 1
                meta["web_results"] += len(results)
                meta["section_citations"][gap.section_title] = [r.url for r in results if getattr(r, "url", None)]
                _trace(f"[NOTES] gap_web_search:done results={len(results)} provider={getattr(cfg.web, 'provider', '')}")
                for idx, res in enumerate(results, start=1):
                    snippet = (res.snippet or "")[: getattr(notes_cfg, "web_snippet_char_limit", 400)]
                    foot_id = f"w_{gap.section_title.lower().replace(' ', '_')}_{idx}"
                    footnote = f"[^{foot_id}]: {res.url} ({res.title})"
                    context_parts.append(f"- {res.title} ({res.url}): {snippet}\n{footnote}")
            if context_parts:
                sec = next((s for s in sections if s["title"] == gap.section_title), None)
                if sec:
                    new_text = _revise_section(sec["text"], gap, "\n".join(context_parts), sec["id"])
                    foots = {line for p in context_parts for line in p.splitlines() if line.startswith("[^")}
                    if foots:
                        new_text = new_text + "\n\n" + "\n".join(foots)
                    patches[sec["id"]] = new_text
                    _trace(f"[NOTES] citations:inserted count={len(context_parts)} section={gap.section_title}")
                    _trace(f"[NOTES] patch_section:applied section={gap.section_title}")
        if patches:
            draft_md = apply_section_patches(draft_md, patches)

    # Round 1 judge
    judge_result = judge_notes(draft_md, cfg, trace=trace, round_num=1)
    current = draft_md
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
