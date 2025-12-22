"""Cleanup utilities for missing assets."""

from __future__ import annotations

from typing import List

from rag_assistant.db import sqlite as db
from rag_assistant.services import asset_service
from rag_assistant.retrieval.vector_store.qdrant import QdrantStore


def list_missing_assets(subject_id: str) -> List[dict]:
    return db.list_assets_with_missing_files(asset_service.get_db_path(), subject_id)


def remove_assets(subject_id: str, asset_ids: List[str], *, remove_vectors: bool = False) -> dict:
    deleted = []
    for asset_id in asset_ids:
        db.delete_asset_dependent_rows(asset_service.get_db_path(), asset_id)
        db.delete_asset(asset_service.get_db_path(), asset_id)
        deleted.append(asset_id)
        if remove_vectors:
            try:
                store = QdrantStore()
                store.delete_by_asset_id(asset_id)
            except Exception:
                # optional best-effort
                pass
    return {"deleted": deleted}
