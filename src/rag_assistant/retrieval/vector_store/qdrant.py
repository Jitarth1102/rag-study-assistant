"""Qdrant vector store helpers."""

from __future__ import annotations

from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_assistant.config import load_config


class QdrantStore:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg
        self.client = QdrantClient(url=cfg.qdrant.url, api_key=None)
        self.collection = cfg.qdrant.collection
        self.vector_size = cfg.embeddings.vector_size if hasattr(cfg, "embeddings") else cfg.qdrant.vector_size
        self.ensure_collection()

    def ensure_collection(self) -> None:
        collections = {c.name: c for c in self.client.get_collections().collections}
        existing = collections.get(self.collection)
        if existing:
            existing_size = existing.vectors_count if hasattr(existing, "vectors_count") else None
            # Fetch vector size from collection info
            info = self.client.get_collection(self.collection)
            current_size = info.vectors_count or getattr(info.vectors, "size", None) or getattr(info.config.params.vectors, "size", None)
            if current_size and int(current_size) != int(self.vector_size):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' has vector size {current_size}, expected {self.vector_size}. "
                    f"Use a new collection name or recreate the collection."
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=self.vector_size, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(self, vectors: List[List[float]], payloads: List[dict], ids: List[str]) -> None:
        if not vectors:
            return
        self.client.upsert(
            collection_name=self.collection,
            points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, vector: List[float], subject_id: str, limit: int) -> List[dict]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            query_filter=qmodels.Filter(must=[qmodels.FieldCondition(key="subject_id", match=qmodels.MatchValue(value=subject_id))]),
        )
        hits = []
        for point in res:
            payload = point.payload or {}
            payload["score"] = point.score
            payload["id"] = point.id
            hits.append(payload)
        return hits

    def delete_by_asset_id(self, asset_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="asset_id", match=qmodels.MatchValue(value=asset_id))]
                ),
            )
        except Exception:
            # best-effort; caller may ignore failures
            pass

    def health_check(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:
            raise RuntimeError(f"Could not reach Qdrant at {self.cfg.qdrant.url}: {exc}")


__all__ = ["QdrantStore"]
