"""PDF rendering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


def render_pdf_to_images(pdf_path: str, out_dir: str, dpi: int) -> List[Dict]:
    """Render PDF pages to PNG images.

    Returns list of metadata dicts: {page_num, image_path, width, height}.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_file)
    results: List[Dict] = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        page_num = page_index + 1
        image_name = f"page_{page_num:04d}.png"
        image_path = output_dir / image_name
        pix.save(image_path)
        results.append(
            {
                "page_num": page_num,
                "image_path": str(image_path),
                "width": pix.width,
                "height": pix.height,
            }
        )
    return results
