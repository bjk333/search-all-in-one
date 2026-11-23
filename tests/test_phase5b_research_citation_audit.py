"""阶段 5B 红灯测试：omni research 自动集成主张级引用对齐审计闭环。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, RequestStats, SearchItem
from omni_search.fetcher import FetchResult
from omni_search.pool import FederatedPool
from omni_search.research import render_research_markdown, run_research_rounds


class _StubResearchEngine:
    id = "stub_res"
    domain = "academic"
    tier = "L0"
    deadline_sec = 5.0

    def __init__(self, items: list[SearchItem]):
        self._items = items

    def is_ready(self) -> tuple[bool, str]:
        return True, "ok"

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=self._items[:limit],
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
        )


@pytest.mark.asyncio
async def test_research_report_produces_citation_audit_and_flags_unsupported() -> None:
    items = [
        SearchItem(
            id="p1",
            domain="academic",
            source="stub_res",
            title="SQLite FTS5 Hybrid Vector Search",
            url="https://arxiv.org/abs/2608.00001",
            snippet="SQLite FTS5 hybrid vector search with BM25F and RRF.",
        ),
        SearchItem(
            id="p2",
            domain="academic",
            source="stub_res",
            title="SQLite FTS5 Hybrid Search Off-Topic Page",
            url="https://other.example.org/offtopic",
            snippet="SQLite FTS5 hybrid search reference.",
        ),
    ]

    async def _mock_fetch(url: str, max_chars: int = 3000, timeout_sec: float = 10.0) -> FetchResult:
        if "arxiv.org" in url:
            return FetchResult(
                ok=True,
                url=url,
                status_code=200,
                title="SQLite FTS5 Hybrid Vector Search",
                content="We implement SQLite FTS5 hybrid vector search combining BM25F ranking and RRF fusion.",
                untrusted_content=True,
            )
        return FetchResult(
            ok=True,
            url=url,
            status_code=200,
            title="Cooking Blog",
            content="Fresh tomato basil pasta recipe with olive oil and garlic boiled for ten minutes.",
            untrusted_content=True,
        )

    pool = FederatedPool(engines=[_StubResearchEngine(items)], cache_ttl_sec=0.0)
    try:
        report = await run_research_rounds(
            pool=pool,
            brief="SQLite FTS5 hybrid vector search",
            domains=["academic"],
            verify_top_k=2,
            render_markdown=True,
            fetcher=_mock_fetch,
        )
    finally:
        await pool.aclose()

    assert report.citation_audit is not None
    assert report.citation_audit.total_checked == 2
    assert report.citation_audit.supported_count == 1
    assert report.citation_audit.unsupported_count == 1
    assert "https://other.example.org/offtopic" in report.citation_audit.flagged_urls
    assert any("citation-unsupported" in g for g in report.gaps)

    md = report.markdown or render_research_markdown(report)
    assert "引用证据对齐审计" in md
    assert "https://other.example.org/offtopic" in md
