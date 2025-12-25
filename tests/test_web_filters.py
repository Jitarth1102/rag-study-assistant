from types import SimpleNamespace

from rag_assistant.web import search_client


class DummyCfg:
    class Web:
        enabled = True
        provider = "serpapi"
        api_key = "key"
        max_results = 5
        timeout_s = 5
        allowed_domains = []
        blocked_domains = []

    web = Web()


def test_allowlist_filters(monkeypatch):
    monkeypatch.setattr(
        search_client,
        "_serpapi_search",
        lambda query, api_key=None, max_results=None, timeout_s=None: [
            search_client.WebResult(title="A", url="http://allow.com/page", snippet="a", source="allow.com"),
            search_client.WebResult(title="B", url="http://block.com/page", snippet="b", source="block.com"),
        ],
    )
    res = search_client.search("q", config=DummyCfg(), allowlist=["allow.com"], blocklist=[])
    assert len(res) == 1
    assert res[0].source == "allow.com"


def test_blocklist_filters(monkeypatch):
    monkeypatch.setattr(
        search_client,
        "_serpapi_search",
        lambda query, api_key=None, max_results=None, timeout_s=None: [
            search_client.WebResult(title="A", url="http://allow.com/page", snippet="a", source="allow.com"),
            search_client.WebResult(title="B", url="http://block.com/page", snippet="b", source="block.com"),
        ],
    )
    res = search_client.search("q", config=DummyCfg(), allowlist=[], blocklist=["block.com"])
    assert len(res) == 1
    assert res[0].source == "allow.com"
