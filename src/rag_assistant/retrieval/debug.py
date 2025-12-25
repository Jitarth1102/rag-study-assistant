from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, List


@dataclass
class RetrievalDebug:
    collection_name: str
    selected_subject_id: str | None
    filter_used: dict | None
    top_k: int
    min_score: float
    query_embedding_dim: int
    query_embedding_min: float
    query_embedding_max: float
    query_embedding_mean: float
    query_embedding_has_nan: bool
    hit_count_raw: int
    hit_count_after_filter: int
    top_hits_preview: List[dict]
    filter_retried_without_subject: bool = False
    judge_reason: str | None = None
    web_queries_attempted: int = 0
    web_queries_used: int = 0
    web_results_used: int = 0
    web_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["RetrievalDebug"]
