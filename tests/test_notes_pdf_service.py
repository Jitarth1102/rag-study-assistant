from rag_assistant.services.notes_pdf_service import render_notes_markdown_to_pdf


def test_render_notes_markdown_to_pdf_basic():
    md = """
# Title

Some paragraph text.

- Bullet one
- Bullet two

1. First
2. Second

```
code block
```
"""
    pdf_bytes = render_notes_markdown_to_pdf(md, title="Sample")
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500
