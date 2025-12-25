"""Shared quality loop for notes generation."""

from __future__ import annotations

import logging
from typing import Optional

from jinja2 import Template

from rag_assistant.llm.provider import generate_answer

logger = logging.getLogger("rag_assistant.notes")


def _log(cfg, message: str) -> None:
    notes_cfg = getattr(cfg, "notes", None)
    if notes_cfg and getattr(notes_cfg, "debug", False):
        logger.info(message)


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
    if params.get("seed") is not None and getattr(getattr(cfg, "llm", None), "provider", "").lower() not in {"ollama"}:
        _trace("[NOTES] warn seed_not_supported provider={}".format(getattr(cfg.llm, "provider", "unknown")))
    critique = generate_answer(_judge_prompt().render(draft=draft_md), cfg, **params)
    critique_text = critique or ""
    needs_revision = not ("no major issues" in critique_text.lower())
    _trace(
        f"[NOTES] judge_review:done round={round_num} items={'na' if not critique_text else len(critique_text.splitlines())} needs_revision={needs_revision}"
    )
    return {"needs_revision": needs_revision, "critique": critique_text}


def run_quality_loop(draft_md: str, config, trace=None) -> str:
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

    # Round 1 judge
    judge_result = judge_notes(draft_md, cfg, trace=trace, round_num=1)
    current = draft_md

    def _revise(text: str, critique_text: str, round_num: int, expand_for_length: bool = False) -> str:
        tag = f"round={round_num}" if not expand_for_length else "expand_for_length"
        _trace(f"[NOTES] revise:start {tag}")
        prompt_text = text
        if expand_for_length:
            prompt_text = (
                text
                + "\n\nThe notes are shorter than desired. Expand missing sections with examples, clarifications, exam tips; avoid fluff."
            )
        draft_for_prompt = prompt_text + ("\n\nCritique:\n" + critique_text if critique_text else "")
        critique_prompt = _critique_prompt().render(draft=draft_for_prompt)
        revised = generate_answer(critique_prompt, cfg, **params)
        _trace(f"[NOTES] revise:done {tag}")
        return revised

    if judge_result.get("needs_revision", True):
        current = _revise(current, judge_result.get("critique", ""), round_num=1)

    # Round 2 judge
    judge_result2 = judge_notes(current, cfg, trace=trace, round_num=2)
    if judge_result2.get("needs_revision", False):
        current = _revise(current, judge_result2.get("critique", ""), round_num=2)

    # Optional expansion only if min_chars > 0 and still short
    if min_chars > 0 and len(current or "") < min_chars:
        current = _revise(current, "", round_num=2, expand_for_length=True)

    return current


__all__ = ["run_quality_loop", "judge_notes"]
