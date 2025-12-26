import fitz

from rag_assistant.services.notes_pdf_service import render_notes_markdown_to_pdf


def test_render_notes_markdown_to_pdf_basic():
    md = """
# Title

Some paragraph text with inline math $a^2 + b^2 = c^2$.

- Bullet one
- Bullet two

1. First
2. Second

```
code block
```

$$E = mc^2$$

Footnote ref[^w1]

[^w1]: https://example.com (Example Site)
"""
    pdf_bytes = render_notes_markdown_to_pdf(md, title="Sample")
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted = "\n".join(page.get_text() for page in doc)
    assert "Title" in extracted
    assert "Bullet one" in extracted
    assert "code block" in extracted
    assert "a^2 + b^2 = c^2" in extracted
    assert "E = mc^2" in extracted
    # Heuristic checks: rendered output should not expose raw markdown markers
    for line in (ln.strip() for ln in extracted.splitlines() if ln.strip()):
        assert not line.startswith("#")
        assert not line.startswith("- ")
        assert not line.startswith("* ")
        assert "```" not in line
        assert "$$" not in line
        assert "[^" not in line
    assert "References" in extracted
