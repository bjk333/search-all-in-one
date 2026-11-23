"""阶段 5C 红灯测试：站内定向采集 safe_crawl_site (map -> fetch_urls_batch) 与 MCP omni_fetch(mode='crawl')。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.mcp_server import OmniMCPServer
from omni_search.site_map import safe_crawl_site


@pytest.mark.asyncio
async def test_safe_crawl_site_maps_ranks_and_fetches_top_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/llms.txt":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=(
                    "# Docs\n"
                    "- [FTS5 Hybrid Guide](https://docs.example.com/fts5-hybrid)\n"
                    "- [Cooking Recipes](https://docs.example.com/cooking)\n"
                    "- [BM25 Ranking Details](https://docs.example.com/bm25-ranking)\n"
                ),
            )
        if p == "/fts5-hybrid":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><head><title>FTS5 Hybrid</title></head><body><main><p>SQLite FTS5 hybrid search combines BM25 and dense vectors.</p></main></body></html>",
            )
        if p == "/bm25-ranking":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><head><title>BM25 Ranking</title></head><body><main><p>BM25F field weighting improves hybrid retrieval accuracy.</p></main></body></html>",
            )
        if p == "/cooking":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><head><title>Cooking</title></head><body><main><p>Pasta and basil.</p></main></body></html>",
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as cli:
        res = await safe_crawl_site(
            url="https://docs.example.com",
            query="FTS5 hybrid BM25",
            max_pages=2,
            max_chars_per_page=2000,
            concurrency=2,
            client=cli,
            resolve_host=False,
        )

    assert res.ok is True
    assert res.untrusted_content is True
    assert res.mapped_total == 3
    assert res.crawled_total == 2
    assert res.succeeded == 2
    crawled_urls = {p.url for p in res.pages}
    assert "https://docs.example.com/fts5-hybrid" in crawled_urls
    assert "https://docs.example.com/bm25-ranking" in crawled_urls
    assert "https://docs.example.com/cooking" not in crawled_urls


@pytest.mark.asyncio
async def test_safe_crawl_site_and_mcp_crawl_mode_block_ssrf() -> None:
    res = await safe_crawl_site(url="http://127.0.0.1:8080/admin", resolve_host=False)
    assert res.ok is False
    assert res.reason_code == "ssrf-blocked"

    srv = OmniMCPServer(env={})
    try:
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "omni_fetch",
                    "arguments": {
                        "url": "http://169.254.169.254/latest/meta-data/",
                        "mode": "crawl",
                        "maxPages": 3,
                    },
                },
            }
        )
        sc = resp["result"]["structuredContent"]
        assert sc["ok"] is False
        assert sc["reason_code"] == "ssrf-blocked"
        assert sc["untrusted_content"] is True
    finally:
        await srv.aclose()
