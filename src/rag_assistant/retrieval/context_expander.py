"""Expand retrieved chunks with neighboring page context."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

from rag_assistant.db.sqlite import execute


def _chunk_identity(chunk: dict) -> str | None:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return chunk_id
    asset = chunk.get("asset_id")
    page = chunk.get("page_num")
    start = chunk.get("start_block")
    if asset is None and page is None and start is None:
        return None
    return f"{asset}:{page}:{start}"


def _load_neighbors_from_db(db_path: Path, asset_id: str, pages: list[int]) -> List[dict]:
    if not pages:
        return []
    placeholders = ",".join(["?"] * len(pages))
    sql = f"SELECT * FROM chunks WHERE asset_id = ? AND page_num IN ({placeholders});"
    params = [asset_id, *pages]
    return execute(db_path, sql, params, fetchall=True) or []


def expand_with_neighbors(chunks: List[dict], *, window: int = 1, max_extra: int = 12, db_path: Path | None = None) -> List[dict]:
    if not chunks or window <= 0 or max_extra <= 0 or db_path is None:
        return chunks

    by_asset_pages = defaultdict(set)
    for ch in chunks:
        by_asset_pages[ch.get("asset_id")].add(ch.get("page_num"))

    neighbor_targets = defaultdict(set)
    for asset_id, pages in by_asset_pages.items():
        for p in pages:
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                neighbor_targets[asset_id].add(p + delta)

    added = []
    existing_ids = set()
    for ch in chunks:
        cid = _chunk_identity(ch)
        if cid:
            existing_ids.add(cid)
    for asset_id, pages in neighbor_targets.items():
        page_list = sorted(p for p in pages if p is not None)
        if not page_list:
            continue
        candidates = _load_neighbors_from_db(db_path, asset_id, page_list)
        # order by closeness to any target page
        def _distance(cand: dict) -> float:
            page_num = cand.get("page_num")
            if page_num is None:
                return float("inf")
            return min(abs(page_num - p) for p in page_list)

        candidates.sort(key=_distance)
        for cand in candidates:
            cid = _chunk_identity(cand)
            if cid in existing_ids:
                continue
            existing_ids.add(cid)
            added.append(cand)
            if len(added) >= max_extra:
                break
        if len(added) >= max_extra:
            break

    return chunks + added


__all__ = ["expand_with_neighbors"]
