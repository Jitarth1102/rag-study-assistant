"""Layout-aware chunker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Dict


def _reading_order(blocks: List[dict]) -> List[dict]:
    return sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))


def _union_bbox(blocks: List[dict]) -> list:
    xs1 = [b["bbox"][0] for b in blocks]
    ys1 = [b["bbox"][1] for b in blocks]
    xs2 = [b["bbox"][2] for b in blocks]
    ys2 = [b["bbox"][3] for b in blocks]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def chunk_ocr_blocks(
    subject_id: str,
    asset_id: str,
    page_num: int,
    ocr_json: dict,
    max_chunk_chars: int,
    min_chunk_chars: int,
    overlap_blocks: int,
) -> List[Dict]:
    blocks = _reading_order(ocr_json.get("blocks", []))
    chunks: List[Dict] = []
    start = 0
    while start < len(blocks):
        current_blocks = []
        char_count = 0
        idx = start
        while idx < len(blocks) and char_count < max_chunk_chars:
            block = blocks[idx]
            text = block.get("text", "")
            current_blocks.append(block)
            char_count += len(text)
            idx += 1
        # ensure min chars if possible
        while idx < len(blocks) and char_count < min_chunk_chars:
            block = blocks[idx]
            current_blocks.append(block)
            char_count += len(block.get("text", ""))
            idx += 1

        if not current_blocks:
            break

        start_block = start
        end_block = start + len(current_blocks) - 1
        chunk_text = "\n".join(b.get("text", "") for b in current_blocks).strip()
        bbox = _union_bbox(current_blocks)
        chunk_id = hashlib.sha256(f"{asset_id}:{page_num}:{start_block}:{end_block}".encode("utf-8")).hexdigest()[:20]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "subject_id": subject_id,
                "asset_id": asset_id,
                "page_num": page_num,
                "text": chunk_text,
                "bbox_json": json.dumps(bbox),
                "start_block": start_block,
                "end_block": end_block,
            }
        )

        if overlap_blocks > 0:
            start = max(idx - overlap_blocks, start + 1)
        else:
            start = idx
    return chunks


def write_chunks_jsonl(chunks: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


__all__ = ["chunk_ocr_blocks", "write_chunks_jsonl"]
