"""Embedding utilities supporting local sentence-transformers and OpenAI fallback."""

from __future__ import annotations

import os
from typing import List, Optional

from rag_assistant.config import load_config

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import
    SentenceTransformer = None  # type: ignore

try:
    from openai import OpenAI, OpenAIError
except Exception:  # pragma: no cover - optional import
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore

_LOCAL_MODEL_CACHE: dict[str, object] = {}


def _get_local_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    if model_name not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _LOCAL_MODEL_CACHE[model_name]


def get_embedding_dim(config=None) -> int:
    cfg = config or load_config()
    return int(cfg.embeddings.vector_size)


class Embedder:
    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.provider = self.cfg.embeddings.provider.lower()
        self.local_model_name = self.cfg.embeddings.model
        self.openai_model = self.cfg.llm.embed_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self._local_model = None
        if self.provider == "openai":
            if not self.api_key or OpenAI is None:
                raise RuntimeError("OPENAI_API_KEY not set or openai package missing")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        else:
            self.client = None

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        model = self._local_model or _get_local_model(self.local_model_name)
        self._local_model = model
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) for v in vecs]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.openai_model, input=texts)
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "local":
            return self._embed_local(texts)
        return self._embed_openai(texts)


__all__ = ["Embedder", "get_embedding_dim"]
