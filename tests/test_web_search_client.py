import types
import pytest

from rag_assistant.web import search_client


class DummyCfg:
    class Web:
        enabled = True
        provider = "serpapi"
        api_key = "key"
        max_results = 5
        timeout_s = 10

    web = Web()


def test_search_success(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert "serpapi.com" in url
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {"organic_results": [{"title": "Result", "link": "http://example.com", "snippet": "Snippet", "source": "example.com"}]},
        )

    monkeypatch.setattr(search_client.requests, "get", fake_get)
    results = search_client.search("test", config=DummyCfg())
    assert results
    assert results[0].url == "http://example.com"


def test_search_missing_key(monkeypatch):
    cfg = DummyCfg()
    cfg.web.api_key = ""
    monkeypatch.setattr(search_client.requests, "get", lambda *a, **k: None)
    with pytest.raises(search_client.WebSearchError):
        search_client.search("test", config=cfg)
