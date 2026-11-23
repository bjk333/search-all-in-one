"""阶段 5A 红灯测试：长查询自适应宽词回退下沉至 FederatedPool。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, RequestStats, SearchItem
from omni_search.pool import FederatedPool, relax_long_query


class _StrictAndEngine:
    """模拟 GitHub/GitLab 严格 AND 搜索：词数 > 4 时 0 命中，词数 <= 4 时命中。"""

    id = "strict_code"
    domain = "code"
    tier = "L0"
    deadline_sec = 5.0

    def __init__(self) -> None:
        self.seen_queries: list[str] = []

    def is_ready(self) -> tuple[bool, str]:
        return True, "ok"

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        self.seen_queries.append(query)
        toks = [t for t in query.split() if t.strip()]
        if len(toks) > 4:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                results=[],
                requests=RequestStats(attempted=1, succeeded=1, failed=0),
            )
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=[
                SearchItem(
                    id="repo-1",
                    domain=self.domain,
                    source=self.id,
                    title="MCP Streamable HTTP Reference",
                    url="https://github.com/modelcontextprotocol/servers",
                    snippet="Official MCP servers with Streamable HTTP and SSE support.",
                )
            ],
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
        )


def test_relax_long_query_extracts_high_signal_terms() -> None:
    q = "how to build an MCP server search fetch extract crawl firecrawl 2026"
    relaxed = relax_long_query(q, max_terms=4)
    toks = relaxed.split()
    assert 2 <= len(toks) <= 4
    # 停用词与泛词应被过滤，保留核心技术词
    low = relaxed.lower()
    assert "how" not in low.split()
    assert "mcp" in low


@pytest.mark.asyncio
async def test_pool_auto_relaxes_long_zero_hit_query_and_discloses_warning() -> None:
    eng = _StrictAndEngine()
    pool = FederatedPool(engines=[eng], cache_ttl_sec=0.0)
    try:
        long_q = "how to build MCP server search fetch extract crawl firecrawl tavily 2026"
        resp = await pool.search(long_q, domains=["code"], limit=5, budget_ms=10000, use_cache=False)
    finally:
        await pool.aclose()

    assert len(eng.seen_queries) == 2
    assert len(eng.seen_queries[1].split()) <= 4
    assert len(resp.results) == 1
    assert resp.results[0].url == "https://github.com/modelcontextprotocol/servers"
    assert resp.ledger.requests_attempted == 2
    assert any("宽词补查" in w for w in resp.warnings)


@pytest.mark.asyncio
async def test_pool_respects_auto_relax_false_and_skips_short_queries() -> None:
    eng = _StrictAndEngine()
    pool = FederatedPool(engines=[eng], cache_ttl_sec=0.0)
    try:
        long_q = "how to build MCP server search fetch extract crawl firecrawl tavily 2026"
        resp = await pool.search(long_q, domains=["code"], limit=5, budget_ms=10000, use_cache=False, auto_relax=False)
    finally:
        await pool.aclose()

    assert len(eng.seen_queries) == 1
    assert len(resp.results) == 0
