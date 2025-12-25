"""Render Markdown notes to PDF using PyMuPDF with lightweight parsing."""

from __future__ import annotations

import io
import re
import textwrap
from typing import List, Tuple

import fitz  # PyMuPDF

try:  # optional math rendering
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

Block = Tuple[str, dict]


MATH_INLINE_RE = re.compile(r"\$(.+?)\$")
MATH_DISPLAY_RE = re.compile(r"^\s*\$\$(.+?)\$\$\s*$")


def _strip_math_tokens(text: str) -> str:
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text)
    text = MATH_INLINE_RE.sub(r"\1", text)
    return text


def _parse_markdown(md: str) -> List[Block]:
    blocks: List[Block] = []
    paragraph: List[str] = []
    code_lines: List[str] = []
    in_code = False
    code_lang = ""
    for raw_line in md.splitlines():
        line = raw_line.rstrip("\n")
        display_math = MATH_DISPLAY_RE.match(line.strip())
        if display_math:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            blocks.append(("math_block", {"text": display_math.group(1).strip()}))
            continue
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line.strip().lstrip("`").strip()
                code_lines = []
            else:
                blocks.append(("code", {"text": "\n".join(code_lines), "lang": code_lang}))
                code_lines = []
                in_code = False
            continue
        if in_code:
            code_lines.append(line)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        bullet_match = re.match(r"^\s*[-*+]\s+(.*)", line)
        ordered_match = re.match(r"^\s*\d+\.\s+(.*)", line)
        hr_match = re.match(r"^\s*(---|\*\*\*|___)\s*$", line)

        if heading_match:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            level = min(len(heading_match.group(1)), 3)
            text = _strip_math_tokens(heading_match.group(2).strip())
            blocks.append(("heading", {"level": level, "text": text}))
        line_no_math = _strip_math_tokens(line)

        if heading_match:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            level = min(len(heading_match.group(1)), 3)
            text = heading_match.group(2).strip()
            blocks.append(("heading", {"level": level, "text": text}))
        elif bullet_match:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            blocks.append(("list_item", {"ordered": False, "text": bullet_match.group(1).strip()}))
        elif ordered_match:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            blocks.append(("list_item", {"ordered": True, "text": ordered_match.group(1).strip()}))
        elif hr_match:
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            blocks.append(("hr", {}))
        elif line.strip().startswith(">"):
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
            text = line.lstrip(">").strip()
            blocks.append(("blockquote", {"text": text}))
        elif line.strip() == "":
            if paragraph:
                blocks.append(("paragraph", {"text": " ".join(paragraph)}))
                paragraph = []
        else:
            paragraph.append(line_no_math.strip())

    if paragraph:
        blocks.append(("paragraph", {"text": " ".join(paragraph)}))
    if code_lines:
        blocks.append(("code", {"text": "\n".join(code_lines), "lang": code_lang}))
    return blocks


def _add_textbox(page, rect, text: str, size: int = 11, font: str = "helv", align: int = 0, color=(0, 0, 0)):
    page.insert_textbox(rect, text, fontsize=size, fontname=font, align=align, color=color)


def render_notes_markdown_to_pdf(markdown_text: str, title: str | None = None) -> bytes:
    """Render Markdown into a PDF and return PDF bytes."""
    doc = fitz.open()
    page = doc.new_page()
    width, height = page.rect.width, page.rect.height
    margin = 54  # 0.75 inch
    cursor_y = margin
    line_spacing = 4

    def new_page():
        nonlocal page, cursor_y
        page = doc.new_page()
        cursor_y = margin

    def ensure_space(needed_height: float):
        nonlocal cursor_y
        if cursor_y + needed_height > height - margin:
            new_page()

    def add_paragraph(text: str, size: int = 11, font: str = "helv", leading: int = 14, indent: int = 0):
        nonlocal cursor_y
        wrap_width = int((width - margin * 2 - indent) / (size * 0.5))
        lines = textwrap.wrap(text, width=max(20, wrap_width)) or [""]
        for line in lines:
            ensure_space(leading)
            page.insert_text(
                (margin + indent, cursor_y),
                line,
                fontsize=size,
                fontname=font,
                fill=(0, 0, 0),
            )
            cursor_y += leading
        cursor_y += line_spacing

    def add_codeblock(text: str):
        nonlocal cursor_y
        code_lines = text.splitlines() or [""]
        max_len = max((len(l) for l in code_lines), default=0)
        approx_height = len(code_lines) * 12 + 10
        ensure_space(approx_height)
        rect = fitz.Rect(margin, cursor_y, width - margin, cursor_y + approx_height)
        page.draw_rect(rect, fill=(0.95, 0.95, 0.95), color=(0.8, 0.8, 0.8))
        cursor_y += 6
        wrap_width = max(20, int((width - margin * 2) / 6))
        wrapped_lines: List[str] = []
        for l in code_lines:
            wrapped_lines.extend(textwrap.wrap(l, width=wrap_width) or [""])
        for line in wrapped_lines:
            page.insert_text((margin + 6, cursor_y), line, fontsize=9, fontname="courier", fill=(0.1, 0.1, 0.1))
            cursor_y += 11
        cursor_y += line_spacing

    def add_mathblock(text: str):
        nonlocal cursor_y
        math_text = text.strip()
        if not math_text:
            return
        ensure_space(40)
        # Always insert readable text; optionally render an image if matplotlib is available.
        page.insert_text(
            (margin, cursor_y),
            math_text,
            fontsize=12,
            fontname="courier",
            fill=(0.1, 0.1, 0.1),
        )
        img_height = 0
        if plt is not None:
            try:
                buf = io.BytesIO()
                fig, ax = plt.subplots(figsize=(3, 0.8), dpi=200)
                ax.axis("off")
                ax.text(0.05, 0.5, f"${math_text}$", fontsize=12)
                fig.tight_layout()
                fig.savefig(buf, format="png", transparent=True)
                plt.close(fig)
                buf.seek(0)
                pix = fitz.Pixmap(buf.getvalue())
                img_height = pix.height / 2.0
                rect = fitz.Rect(margin, cursor_y + 14, margin + pix.width / 2.0, cursor_y + 14 + img_height)
                page.insert_image(rect, stream=buf.getvalue())
            except Exception:
                pass
        cursor_y += max(24, img_height + 18)
        cursor_y += line_spacing

    def add_hr():
        nonlocal cursor_y
        ensure_space(10)
        page.draw_line((margin, cursor_y), (width - margin, cursor_y), color=(0.6, 0.6, 0.6))
        cursor_y += line_spacing + 4

    blocks = _parse_markdown(markdown_text or "")

    if title:
        add_paragraph(title, size=16, font="helv", leading=18)

    list_counter = 1
    for kind, data in blocks:
        if kind == "heading":
            lvl = data.get("level", 1)
            size = 16 if lvl == 1 else 14 if lvl == 2 else 12
            leading = size + 2
            add_paragraph(data.get("text", ""), size=size, font="helv", leading=leading)
        elif kind == "paragraph":
            add_paragraph(data.get("text", ""), size=11, font="helv", leading=14)
        elif kind == "list_item":
            prefix = f"{list_counter}." if data.get("ordered") else "•"
            text = f"{prefix} {data.get('text', '')}"
            add_paragraph(text, size=11, font="helv", leading=14, indent=12)
            if data.get("ordered"):
                list_counter += 1
            else:
                list_counter = 1
        elif kind == "code":
            add_codeblock(data.get("text", ""))
        elif kind == "math_block":
            add_mathblock(data.get("text", ""))
        elif kind == "blockquote":
            add_paragraph(data.get("text", ""), size=11, font="helv", leading=14, indent=18)
        elif kind == "hr":
            add_hr()
        else:
            add_paragraph(str(data), size=11, font="helv", leading=14)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


__all__ = ["render_notes_markdown_to_pdf"]
