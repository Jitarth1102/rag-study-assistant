"""Domain models for the assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Citation:
    source: str
    page: Optional[int] = None
    quote: Optional[str] = None


@dataclass
class ChatResponse:
    answer: str
    citations: List[Citation]
