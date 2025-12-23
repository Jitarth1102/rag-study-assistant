"""Qdrant vector store helpers."""

from __future__ import annotations

from typing import List
from types import SimpleNamespace

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_assistant.config import load_config
from rag_assistant.vectorstore.point_id import make_point_uuid


def _normalize_points(points) -> list:
    normalized = []
    for pt in points or []:
        payload = getattr(pt, "payload", None)
        score = getattr(pt, "score", None)
        pid = getattr(pt, "id", None)
        if isinstance(pt, dict):
            payload = pt.get("payload", payload)
            score = pt.get("score", score)
            pid = pt.get("id", pid)
        normalized.append(SimpleNamespace(id=pid, score=score, payload=payload))
    return normalized


def search_points(client, collection_name: str, vector: list[float], limit: int, *, query_filter=None, with_payload: bool = True):
    """Compatibility wrapper for different qdrant-client APIs."""
    if hasattr(client, "search"):
        res = client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=with_payload,
            query_filter=query_filter,
        )
        return _normalize_points(res)
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=with_payload,
            query_filter=query_filter,
        )
        return _normalize_points(res)
    raise RuntimeError("Unsupported Qdrant client version: missing search/query_points")


class QdrantStore:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg
        self.client = QdrantClient(url=cfg.qdrant.url, api_key=None)
        self.collection = cfg.qdrant.collection
        self.vector_size = cfg.embeddings.vector_size if hasattr(cfg, "embeddings") else cfg.qdrant.vector_size
        self.ensure_collection()

    def get_collection_point_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
        except Exception as exc:
            raise RuntimeError(f"Could not fetch collection info for {self.collection}: {exc}")
        for attr in ("points_count", "vectors_count"):
            val = getattr(info, attr, None)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    continue
        return 0

    def ensure_collection(self) -> None:
        collections = {c.name: c for c in self.client.get_collections().collections}
        existing = collections.get(self.collection)
        if existing:
            # Fetch vector size from collection info
            info = self.client.get_collection(self.collection)
            current_size = getattr(info, "points_count", None)  # placeholder to trigger fetch
            current_size = getattr(info, "vectors_count", None) or getattr(
                getattr(info, "vectors", None), "size", None
            ) or getattr(getattr(info, "config", None), "params", None)
            if current_size and hasattr(current_size, "vectors"):
                current_size = getattr(current_size.vectors, "size", None)
            if current_size is None:
                current_size = getattr(info, "points_count", None)
            current_size = int(current_size) if current_size else None
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
        res = search_points(
            self.client,
            collection_name=self.collection,
            vector=vector,
            limit=limit,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="subject_id", match=qmodels.MatchValue(value=subject_id))]
            ),
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


__all__ = ["QdrantStore", "search_points"]
