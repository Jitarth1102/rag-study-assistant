"""Qdrant vector store helpers."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import List
from types import SimpleNamespace
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute

logger = logging.getLogger(__name__)


def _extract_points(result) -> list:
    if result is None:
        return []
    res = result
    if isinstance(res, dict) and "result" in res:
        res = res["result"]
    if hasattr(res, "result"):
        res = res.result
    if isinstance(res, dict) and "points" in res:
        res = res["points"]
    if hasattr(res, "points"):
        res = res.points
    if isinstance(res, (list, tuple)):
        return list(res)
    return [res]


def _normalize_points(result) -> list:
    points = _extract_points(result)
    normalized = []
    for idx, pt in enumerate(points):
        payload = getattr(pt, "payload", None)
        score = getattr(pt, "score", None)
        pid = getattr(pt, "id", None)
        if isinstance(pt, dict):
            payload = pt.get("payload", payload)
            score = pt.get("score", score)
            pid = pt.get("id", pid)
        if payload is None:
            payload = {}
        if os.getenv("QDRANT_DEBUG") and idx == 0:
            logger.info(
                "Qdrant raw hit debug",
                extra={
                    "type": type(pt).__name__,
                    "attrs": [a for a in dir(pt) if not a.startswith("_")] if not isinstance(pt, dict) else list(pt.keys()),
                    "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
                },
            )
        normalized.append(SimpleNamespace(id=pid, score=score, payload=payload))
    return normalized


def search_points(client, collection_name: str, vector: list[float], limit: int, *, query_filter=None, with_payload: bool = True):
    """Compatibility wrapper for different qdrant-client APIs."""
    if hasattr(client, "search"):
        try:
            res = client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=with_payload,
                query_filter=query_filter,
            )
        except TypeError:
            # older clients use "filter"
            res = client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=with_payload,
                filter=query_filter,
            )
        return _normalize_points(res)
    if hasattr(client, "query_points"):
        sig = inspect.signature(client.query_points)
        filter_key = "query_filter" if "query_filter" in sig.parameters else "filter"
        query_key = "query_vector" if "query_vector" in sig.parameters else "query"
        kwargs = {
            "collection_name": collection_name,
            query_key: vector,
            "limit": limit,
            "with_payload": with_payload,
            filter_key: query_filter,
        }
        res = client.query_points(**kwargs)
        return _normalize_points(res)
    raise RuntimeError("Unsupported Qdrant client version: missing search/query_points")


class QdrantStore:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg
        self.client = QdrantClient(url=cfg.qdrant.url, api_key=None)
        self.collection = cfg.qdrant.collection
        self.vector_size = cfg.embeddings.vector_size if hasattr(cfg, "embeddings") else cfg.qdrant.vector_size
        self.db_path = Path(cfg.database.sqlite_path)
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
            info = self.client.get_collection(self.collection)
            current_size = self._get_collection_vector_size(info)
            if current_size is None:
                logger.warning(
                    "Could not determine vector size for existing Qdrant collection '%s'; skipping compatibility check.",
                    self.collection,
                )
                return
            if current_size != int(self.vector_size):
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

    def _hydrate_payload(self, payload: dict | None) -> dict:
        payload = payload.copy() if payload else {}
        if payload.get("text"):
            return payload
        chunk_id = payload.get("chunk_id")
        if chunk_id:
            row = execute(
                self.db_path,
                "SELECT text, page_num, asset_id, subject_id, bbox_json, start_block, end_block FROM chunks WHERE chunk_id = ?;",
                (chunk_id,),
                fetchone=True,
            )
            if row:
                payload.setdefault("text", row.get("text"))
                payload.setdefault("page_num", row.get("page_num"))
                payload.setdefault("asset_id", row.get("asset_id"))
                payload.setdefault("subject_id", row.get("subject_id"))
                payload.setdefault("bbox_json", row.get("bbox_json"))
                payload.setdefault("start_block", row.get("start_block"))
                payload.setdefault("end_block", row.get("end_block"))
                payload.setdefault("page_num", row.get("page_num"))
        return payload

    @staticmethod
    def _maybe_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _vector_size_from_vectors_cfg(cls, vectors_cfg) -> int | None:
        if vectors_cfg is None:
            return None
        root = getattr(vectors_cfg, "__root__", None)
        if root is not None:
            return cls._vector_size_from_vectors_cfg(root)
        size = getattr(vectors_cfg, "size", None)
        if size is not None:
            return cls._maybe_int(size)
        if isinstance(vectors_cfg, dict):
            cfg = vectors_cfg.get("default") or next(iter(vectors_cfg.values()), None)
            if cfg is None:
                return None
            root_cfg = getattr(cfg, "__root__", None)
            if root_cfg is not None:
                cfg = root_cfg
            if isinstance(cfg, dict):
                return cls._maybe_int(cfg.get("size") or cfg.get("vector_size"))
            return cls._maybe_int(getattr(cfg, "size", None))
        nested = getattr(vectors_cfg, "vectors", None)
        if nested is not None:
            return cls._vector_size_from_vectors_cfg(nested)
        return None

    @classmethod
    def _get_collection_vector_size(cls, info) -> int | None:
        config = getattr(info, "config", None)
        params = getattr(config, "params", None) if config is not None else None
        vectors_cfg = getattr(params, "vectors", None) if params is not None else None
        size = cls._vector_size_from_vectors_cfg(vectors_cfg)
        if size is not None:
            return size
        # Some clients expose vectors directly
        size = cls._vector_size_from_vectors_cfg(getattr(info, "vectors", None))
        if size is not None:
            return size
        # Older fallback
        if params is not None:
            return cls._maybe_int(getattr(params, "vector_size", None) or getattr(params, "size", None))
        return None

    def search(self, vector: List[float], subject_id: str | None, limit: int) -> List[dict]:
        def _search(filter_obj):
            return search_points(
                self.client,
                collection_name=self.collection,
                vector=vector,
                limit=limit,
                query_filter=filter_obj,
                with_payload=True,
            )

        filter_obj = (
            qmodels.Filter(must=[qmodels.FieldCondition(key="subject_id", match=qmodels.MatchValue(value=subject_id))])
            if subject_id
            else None
        )
        res = _search(filter_obj)

        def _process(res_points):
            hits_local = []
            for point in res_points:
                raw_payload = getattr(point, "payload", {}) or {}
                if not isinstance(raw_payload, dict):
                    try:
                        raw_payload = dict(raw_payload)  # type: ignore[arg-type]
                    except Exception:
                        raw_payload = {}
                payload = self._hydrate_payload(raw_payload)
                payload.setdefault("chunk_id", raw_payload.get("chunk_id"))
                payload.setdefault("page_num", raw_payload.get("page_num"))
                payload.setdefault("asset_id", raw_payload.get("asset_id"))
                payload.setdefault("subject_id", raw_payload.get("subject_id"))
                chunk_id = payload.get("chunk_id")
                if not chunk_id:
                    logger.warning("Dropping Qdrant hit without chunk_id", extra={"point_id": getattr(point, "id", None)})
                    continue
                if not payload.get("text"):
                    payload = self._hydrate_payload(payload)
                if not payload.get("text") or payload.get("page_num") is None:
                    logger.warning("Dropping Qdrant hit missing text/page_num after hydration", extra={"chunk_id": chunk_id})
                    continue
                payload["score"] = payload.get("score", getattr(point, "score", None))
                payload["id"] = getattr(point, "id", None)
                hits_local.append(payload)
            return hits_local

        hits = _process(res)
        if not hits and subject_id:
            retry_res = _search(None)
            hits = _process(retry_res)
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


__all__ = ["QdrantStore", "search_points", "_normalize_points"]
