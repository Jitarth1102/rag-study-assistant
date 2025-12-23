from pathlib import Path

import fitz

from rag_assistant.ingest.render.pdf_to_images import render_pdf_to_images


def test_render_pdf_to_images(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((50, 100), "Hello")
    doc.save(pdf_path)
    doc.close()

    out_dir = tmp_path / "out"
    pages = render_pdf_to_images(str(pdf_path), str(out_dir), dpi=72)
    assert len(pages) == 1
    assert Path(pages[0]["image_path"]).exists()
