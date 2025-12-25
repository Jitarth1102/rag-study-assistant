"""LLM provider dispatcher."""

from __future__ import annotations

import os

from rag_assistant.config import load_config
from rag_assistant.llm.ollama_client import OllamaClient, OllamaError
from rag_assistant.retrieval.embedder import Embedder  # noqa: F401  # embedder still uses OpenAI for now

try:
    from openai import OpenAI
    from openai import OpenAIError
except Exception:  # pragma: no cover - optional import
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore


def generate_answer(prompt: str, config=None, **generation_kwargs) -> str:
    cfg = config or load_config()
    provider = cfg.llm.provider.lower()
    temperature = generation_kwargs.get("temperature", cfg.llm.temperature)
    top_p = generation_kwargs.get("top_p")
    seed = generation_kwargs.get("seed")
    max_tokens = generation_kwargs.get("max_tokens")
    if provider == "ollama":
        client = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model, timeout_s=cfg.llm.timeout_s)
        return client.generate(prompt, temperature=temperature, top_p=top_p, seed=seed, max_tokens=max_tokens)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or OpenAI is None:
            return "OPENAI_API_KEY not set; cannot generate answer."
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        completion = client.chat.completions.create(
            model=cfg.llm.chat_model,
            messages=[{"role": "system", "content": "Answer using only provided notes."}, {"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p or 1,
            max_tokens=max_tokens or 400,
        )
        return completion.choices[0].message.content or ""
    return "LLM provider not supported."


__all__ = ["generate_answer"]
