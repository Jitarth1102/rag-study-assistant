"""Minimal Ollama client for local LLM generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests


class OllamaError(RuntimeError):
    """Raised when Ollama generation fails."""


@dataclass
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    timeout_s: int = 60

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        if max_tokens is not None:
            payload["num_predict"] = max_tokens
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:  # pragma: no cover - network path
            raise OllamaError(f"Ollama not reachable at {self.base_url}. Start Ollama with 'ollama serve'. Details: {exc}")
        if resp.status_code != 200:
            raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Invalid JSON from Ollama: {resp.text}") from exc
        output = data.get("response") or data.get("text") or ""
        if not output:
            raise OllamaError("Empty response from Ollama.")
        return output


__all__ = ["OllamaClient", "OllamaError"]
