"""Vector store base interfaces."""

from __future__ import annotations

from typing import Protocol, List, Any


class VectorStore(Protocol):
    def add(self, items: List[Any]) -> None: ...

    def search(self, query: str, limit: int = 5) -> list[Any]: ...
