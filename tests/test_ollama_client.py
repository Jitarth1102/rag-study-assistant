import json

import pytest
import requests

from rag_assistant.llm.ollama_client import OllamaClient, OllamaError


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_ollama_client_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(200, {"response": "hello"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(base_url="http://127.0.0.1:11434", model="llama3.1:8b", timeout_s=10)
    out = client.generate("prompt", temperature=0.2)
    assert out == "hello"
    assert captured["json"]["model"] == "llama3.1:8b"
    assert captured["json"]["stream"] is False


def test_ollama_client_connection_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient()
    with pytest.raises(OllamaError):
        client.generate("prompt")
