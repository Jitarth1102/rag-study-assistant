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
    notes_ids: List[str] = []
    for asset_id in asset_ids:
        # capture notes ids for cleanup
        rows = db.execute(
            asset_service.get_db_path(),
            "SELECT notes_id FROM notes WHERE asset_id = ?;",
            (asset_id,),
            fetchall=True,
        )
        notes_ids.extend([r["notes_id"] for r in rows or []])
        db.delete_asset_dependent_rows(asset_service.get_db_path(), asset_id)
        db.delete_asset(asset_service.get_db_path(), asset_id)
        deleted.append(asset_id)
        if remove_vectors:
            try:
                store = QdrantStore()
                store.delete_by_asset_id(asset_id)
                for nid in notes_ids:
                    store.delete_by_notes_id(nid)
            except Exception:
                # optional best-effort
                pass
    return {"deleted": deleted}
