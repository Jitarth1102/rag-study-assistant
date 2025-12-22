"""Qdrant client stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag_assistant.config import load_config


@dataclass
class QdrantClientStub:
    host: str
    port: int
    collection_name: str

    def is_running(self) -> bool:
        # Placeholder health check hook
        return True

    def add(self, items: List[Any]) -> None:  # pragma: no cover - stub
        raise NotImplementedError("Vector add not implemented yet")

    def search(self, query: str, limit: int = 5) -> list[Any]:  # pragma: no cover - stub
        raise NotImplementedError("Vector search not implemented yet")


def get_qdrant_client() -> QdrantClientStub:
    cfg = load_config()
    return QdrantClientStub(host=cfg.qdrant.host, port=cfg.qdrant.port, collection_name=cfg.qdrant.collection_name)


__all__ = ["QdrantClientStub", "get_qdrant_client"]
