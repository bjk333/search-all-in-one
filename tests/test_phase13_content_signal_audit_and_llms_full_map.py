"""阶段 13 回归测试：site_map /llms-full.txt 自动发现 + omni crawl / omni research Content-Signal 限制策略审计披露。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, RequestStats, SearchItem
from omni_search.fetcher import FetchResult
from omni_search.pool import FederatedPool
from omni_search.research import run_research_rounds
from omni_search.site_map import safe_crawl_site, safe_map_site


class _FakeEngine:
    def __init__(self) -> None:
        self.id = "fake_web"
        self.domain = "web"
        self.tier = "L0"
        self.deadline_sec = 5.0

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            elapsed_ms=5,
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
            results=[
                SearchItem(
                    id="r1",
                    domain="web",
                    source=self.id,
                    title="Restricted Publisher Spec",
                    url="https://policy.example.com/spec",
                    snippet="Agentic web content signals specification.",
                    score=0.95,
                )
            ],
        )


@pytest.mark.asyncio
async def test_site_map_llms_full_fallback_and_crawl_content_signal_warning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://docs.example.com/llms.txt":
            return httpx.Response(404, text="not found")
        if url == "https://docs.example.com/llms-full.txt":
            return httpx.Response(
                200,
                text="# Full Index\n- [API Guide](https://docs.example.com/api)\n- [Policy Page](https://docs.example.com/policy)\n",
                headers={"content-type": "text/markdown"},
            )
        if url == "https://docs.example.com/api":
            return httpx.Response(
                200,
                text="# API Guide\n\nOpen documentation for agents.",
                headers={"content-type": "text/markdown", "content-signal": "search=yes, ai-input=yes"},
            )
        if url == "https://docs.example.com/policy":
            return httpx.Response(
                200,
                text="# Policy Page\n\nRestricted for AI input.",
                headers={"content-type": "text/markdown", "content-signal": "search=yes, ai-input=no"},
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        mres = await safe_map_site("https://docs.example.com", client=cli, resolve_host=False)
        assert mres.ok is True
        assert "llms-full-txt" in mres.sources_used
        assert {e.url for e in mres.urls} == {
            "https://docs.example.com/api",
            "https://docs.example.com/policy",
        }

        cres = await safe_crawl_site(
            "https://docs.example.com",
            max_pages=4,
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert cres.ok is True
        assert cres.succeeded == 2
        assert any("content-signal-restricted" in w and "ai-input=no" in w for w in cres.warnings)
        assert not any("/api" in w for w in cres.warnings)


@pytest.mark.asyncio
async def test_research_verify_records_content_signal_restriction_gap() -> None:
    pool = FederatedPool([_FakeEngine()])

    async def fake_fetch(url: str, max_chars: int = 3000, timeout_sec: float = 10.0) -> FetchResult:
        return FetchResult(
            ok=True,
            url=url,
            final_url=url,
            status_code=200,
            content_type="text/markdown",
            source_via="markdown-negotiated",
            title="Restricted Publisher Spec",
            content="# Restricted Publisher Spec\n\nAgentic web content signals specification details.",
            content_signals={"search": "no", "ai-input": "no"},
            markdown_tokens=240,
            untrusted_content=True,
        )

    rep = await run_research_rounds(
        pool,
        "agentic web content signals",
        domains=["web"],
        mode="quick",
        verify_top_k=1,
        fetcher=fake_fetch,
    )
    assert any("content-signal-restricted" in g and "ai-input=no" in g and "search=no" in g for g in rep.gaps)
