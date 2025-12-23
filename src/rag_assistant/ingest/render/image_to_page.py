"""Handle single image assets as single-page documents."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import cv2


def normalize_image_to_page(image_path: str, out_dir: str) -> Dict:
    """Copy a standalone image into the pages directory as page_0001.png."""
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "page_0001.png"
    shutil.copyfile(src, target)

    img = cv2.imread(str(target))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    height, width = img.shape[:2]

    return {"page_num": 1, "image_path": str(target), "width": width, "height": height}
