"""Web search client (SerpAPI) with safe defaults and clear results."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import List, Optional

import requests
from urllib.parse import urlparse

from rag_assistant.config import load_config


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source: str
    published_at: Optional[str] = None


class WebSearchError(RuntimeError):
    """Raised when web search fails."""


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or ""
        return host.lower()
    except Exception:
        return ""


def _filter_results(results: List[WebResult], allowlist: list[str], blocklist: list[str]) -> List[WebResult]:
    allow = [d.lower().strip() for d in allowlist if d] if allowlist else []
    block = [d.lower().strip() for d in blocklist if d] if blocklist else []
    filtered = []
    for res in results:
        domain = res.source.lower() if res.source else _extract_domain(res.url)
        if block and any(domain.endswith(b) for b in block):
            continue
        if allow and not any(domain.endswith(a) for a in allow):
            continue
        filtered.append(res)
    return filtered


def _serpapi_search(query: str, api_key: str, max_results: int, timeout_s: int) -> List[WebResult]:
    if not api_key:
        raise WebSearchError("SerpAPI key missing; set WEB_API_KEY or web.api_key.")
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max_results,
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=timeout_s)
    except (requests.RequestException, socket.timeout) as exc:
        raise WebSearchError(f"SerpAPI request failed: {exc}") from exc
    if resp.status_code != 200:
        raise WebSearchError(f"SerpAPI returned {resp.status_code}: {resp.text}")
    try:
        data = resp.json()
    except Exception as exc:
        raise WebSearchError(f"Invalid JSON from SerpAPI: {resp.text}") from exc

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        title = item.get("title") or ""
        url = item.get("link") or item.get("url") or ""
        snippet = item.get("snippet") or ""
        source = item.get("source") or _extract_domain(url)
        results.append(WebResult(title=title, url=url, snippet=snippet, source=source))
    return results


def search(query: str, max_results: Optional[int] = None, config=None, allowlist: Optional[list[str]] = None, blocklist: Optional[list[str]] = None) -> List[WebResult]:
    cfg = config or load_config()
    web_cfg = cfg.web
    if not web_cfg.enabled:
        raise WebSearchError("Web search is disabled. Enable via WEB_ENABLED=true or web.enabled.")

    provider = (web_cfg.provider or "serpapi").lower()
    max_res = max_results or web_cfg.max_results
    timeout_s = web_cfg.timeout_s
    env_api_key = os.getenv("WEB_API_KEY")
    api_key = env_api_key if env_api_key is not None else web_cfg.api_key

    if provider != "serpapi":
        raise WebSearchError(f"Unsupported web provider: {provider}")

    results = _serpapi_search(query, api_key=api_key, max_results=max_res, timeout_s=timeout_s)
    return _filter_results(results, allowlist or [], blocklist or [])


__all__ = ["search", "WebResult", "WebSearchError"]
