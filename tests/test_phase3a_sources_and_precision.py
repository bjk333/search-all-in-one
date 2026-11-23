"""Phase 3A 红灯测试：50 原生引擎（PyPI / Go / HF Papers / OSV.dev）、PyPI 官方 API 直取与泛词防霸榜精修。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import CompletenessReport, LedgerSummary, SearchItem, SearchResponse, build_coverage
from omni_search.fetcher import safe_fetch_url
from omni_search.mcp_server import build_default_engines
from omni_search.research import _merge_lane_candidates, _select_diverse_verify_targets, widen_query


def test_new_declarative_yaml_sources_and_50_engine_count():
    """原生引擎总数应升级至 50（44 声明式 + 6 Python 原生），补齐 PyPI/Go/HF Papers/OSV.dev。"""
    engines = build_default_engines(env={})
    ids = {e.id for e in engines}
    assert len(ids) == 50
    assert {"pypi_ecosystems", "golang_ecosystems", "hf_papers", "osv_vulns"}.issubset(ids)
    sec_ids = {e.id for e in engines if e.domain == "security"}
    assert sec_ids == {"nvd_cve", "osv_vulns"}


@pytest.mark.asyncio
async def test_osv_vulns_engine_by_id_and_by_package():
    """OsvVulnsEngine 应支持漏洞 ID 直查、带生态前缀包查以及无前缀多生态批量查。"""
    from omni_search.security_engines import OsvVulnsEngine

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if u == "https://api.osv.dev/v1/vulns/GHSA-j8r2-6x86-q33q":
            return httpx.Response(
                200,
                json={
                    "id": "GHSA-j8r2-6x86-q33q",
                    "aliases": ["CVE-2023-32681"],
                    "summary": "Requests leaks Proxy-Authorization headers",
                    "published": "2023-05-26T00:00:00Z",
                },
            )
        if u == "https://api.osv.dev/v1/query":
            return httpx.Response(
                200,
                json={
                    "vulns": [
                        {
                            "id": "GHSA-2xpw-w6gg-jr37",
                            "aliases": ["CVE-2026-21441"],
                            "summary": "urllib3 streaming decompression issue",
                            "published": "2026-01-07T00:00:00Z",
                        }
                    ]
                },
            )
        if u == "https://api.osv.dev/v1/querybatch":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"vulns": [{"id": "GHSA-2xpw-w6gg-jr37", "modified": "2026-09-25T17:15:05Z"}]},
                        {},
                        {},
                        {},
                    ]
                },
            )
        return httpx.Response(404)

    eng = OsvVulnsEngine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        r_id = await eng.search("GHSA-j8r2-6x86-q33q", limit=3, client=cli)
        assert r_id.failure is None
        assert len(r_id.results) == 1
        assert "CVE-2023-32681" in r_id.results[0].snippet
        assert r_id.results[0].url == "https://osv.dev/vulnerability/GHSA-j8r2-6x86-q33q"

        r_pkg = await eng.search("pypi:urllib3", limit=3, client=cli)
        assert r_pkg.failure is None
        assert len(r_pkg.results) == 1
        assert "GHSA-2xpw-w6gg-jr37" in r_pkg.results[0].title

        r_auto = await eng.search("urllib3 vulnerability", limit=3, client=cli)
        assert r_auto.failure is None
        assert len(r_auto.results) == 1
        assert r_auto.results[0].id == "GHSA-2xpw-w6gg-jr37"


@pytest.mark.asyncio
async def test_fetcher_pypi_json_api_and_research_broad_terms():
    """验证 PyPI 官方 JSON API 直取，以及泛词 research/learning 不再让 MDN 泛文抢占核验位。"""

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url) == "https://pypi.org/pypi/httpx/json":
            return httpx.Response(
                200,
                json={
                    "info": {
                        "name": "httpx",
                        "version": "0.28.1",
                        "summary": "The next generation HTTP client.",
                        "description": "# HTTPX\n\nA fully featured async HTTP client for Python 3.",
                    }
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url("https://pypi.org/project/httpx/", client=cli, resolve_host=False)
        assert res.ok is True
        assert res.source_via == "pypi-json-api"
        assert "0.28.1" in res.content
        assert "async HTTP client" in res.content

    brief = "deep research agentic search contextual retrieval late chunking 2026"
    wq = widen_query(brief, max_terms=4).lower()
    assert "research" not in wq
    assert "contextual" in wq and "chunking" in wq

    mdn = SearchItem(
        id="m1",
        domain="knowledge",
        source="mdn",
        title="Research and learning - Learn web development | MDN",
        url="https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Soft_skills/Research_and_learning",
        snippet="This article provides useful tips on research and learning.",
        score=0.03,
    )
    paper1 = SearchItem(
        id="p1",
        domain="academic",
        source="hf_papers",
        title="Agentic Contextual Retrieval and Late Chunking",
        url="https://arxiv.org/abs/2601.11111",
        snippet="Evaluates contextual retrieval and late chunking for agentic search.",
        score=0.018,
    )
    paper2 = SearchItem(
        id="p2",
        domain="code",
        source="github_direct",
        title="contextual-chunking-rag",
        url="https://github.com/org/contextual-chunking-rag",
        snippet="Open-source implementation of contextual retrieval and late chunking.",
        score=0.017,
    )
    resp = SearchResponse(
        version="0.1.0",
        query=brief,
        domains=["knowledge", "academic", "code"],
        results=[mdn, paper1, paper2],
        coverage=build_coverage([]),
        completeness=CompletenessReport(returned_count=3, total_candidates=3, truncated=False, per_source_limit=5),
        ledger=LedgerSummary(run_id="t3a", elapsed_ms=10, budget_ms=1000),
    )
    merged = _merge_lane_candidates([("overview", resp)], brief=brief)
    picked = _select_diverse_verify_targets(merged, top_k=2)
    assert all(c.url != mdn.url for c in picked)
