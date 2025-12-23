"""Utilities for deterministic Qdrant point IDs."""

from __future__ import annotations

import uuid

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def make_point_uuid(identity: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, identity))


__all__ = ["make_point_uuid"]
