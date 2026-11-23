"""Phase 2A 红灯测试：智能提词、相关性重排防 T1 霸榜、Fast-Fail 分级熔断、Readability 清洗与 fetch 缓存。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.cache import DiskCacheStore
from omni_search.contracts import (
    CompletenessReport,
    EngineEnvelope,
    EngineFailure,
    LedgerSummary,
    ReasonCode,
    SearchItem,
    SearchResponse,
    build_coverage,
)
from omni_search.fetcher import safe_fetch_url
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool
from omni_search.research import (
    _merge_lane_candidates,
    _select_diverse_verify_targets,
    plan_research_lanes,
    widen_query,
)


def test_widen_query_preserves_high_information_technical_terms():
    """长查询提词应保留高区分度技术实体（MCP/markdown/reranking、sqlite/fts5/markitdown/docling），而非按词序盲切。"""
    q1 = widen_query("AI agent web search MCP server markdown extraction reranking", max_terms=4)
    q1_low = q1.lower()
    assert "mcp" in q1_low
    assert "markdown" in q1_low
    assert "reranking" in q1_low

    q2 = widen_query("local hybrid search sqlite vector fts5 document parser markitdown docling", max_terms=4)
    q2_low = q2.lower()
    assert "fts5" in q2_low
    assert "markitdown" in q2_low or "docling" in q2_low
    assert "sqlite" in q2_low

    # 短查询仍完整保留
    q3 = widen_query("sqlite hybrid search", max_terms=3)
    assert set(q3.lower().split()) == {"sqlite", "hybrid", "search"}

    plan = plan_research_lanes("local hybrid search sqlite vector fts5 markitdown", mode="feasibility")
    alt = next(l for l in plan.lanes if l.name == "alternatives")
    assert "fts5" in alt.query.lower() or "sqlite" in alt.query.lower()


def test_merge_lane_candidates_prevents_low_relevance_t1_domination():
    """低相关性 T1 泛词条（如 MDN search.search()）不得凭借 T1 权重压过高度切题的候选，也不得抢占核验名额。"""
    brief = "local hybrid search sqlite vector fts5 document parser markitdown"

    mdn_item = SearchItem(
        id="mdn1",
        domain="knowledge",
        source="mdn",
        title="search.search() - WebExtensions API",
        url="https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/search/search",
        snippet="Perform a browser search using the default browser search engine.",
        score=0.025,
    )
    relevant_item = SearchItem(
        id="gh1",
        domain="code",
        source="github_direct",
        title="otzar - Local hybrid search over SQLite FTS5 and vector embeddings",
        url="https://github.com/danielfleischer/otzar",
        snippet="Local hybrid search combining SQLite FTS5 BM25, dense vector similarity, and markdown document parser.",
        score=0.018,
    )
    paper_item = SearchItem(
        id="arx1",
        domain="academic",
        source="arxiv",
        title="SQLite is Enough: Hybrid Lexical and Vector Search with FTS5",
        url="https://arxiv.org/abs/2608.24060",
        snippet="Hybrid search within SQLite using FTS5 and vector search.",
        score=0.017,
    )

    resp1 = SearchResponse(
        version="0.1.0",
        query=brief,
        domains=["knowledge", "code"],
        results=[mdn_item, relevant_item],
        coverage=build_coverage([]),
        completeness=CompletenessReport(returned_count=2, total_candidates=2, truncated=False, per_source_limit=5),
        ledger=LedgerSummary(run_id="t1", elapsed_ms=10, budget_ms=1000),
    )
    resp2 = SearchResponse(
        version="0.1.0",
        query=brief,
        domains=["knowledge", "academic"],
        results=[mdn_item, paper_item],
        coverage=build_coverage([]),
        completeness=CompletenessReport(returned_count=2, total_candidates=2, truncated=False, per_source_limit=5),
        ledger=LedgerSummary(run_id="t2", elapsed_ms=10, budget_ms=1000),
    )

    merged = _merge_lane_candidates([("overview", resp1), ("academic", resp2)], brief=brief)
    assert merged[0].url != mdn_item.url
    assert merged[-1].url == mdn_item.url

    picked = _select_diverse_verify_targets(merged, top_k=2)
    assert all(c.url != mdn_item.url for c in picked)


@pytest.mark.asyncio
async def test_health_store_fast_fail_for_quota_and_repeated_failures(tmp_path):
    """quota-exhausted 与连续 >=4 次超时应在冷却期内直接 Fast-Fail（0 次网络请求），且支持 reset。"""
    store = HealthStore(tmp_path / "health.json")
    store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=1000.0)
    st_q = store.check("deepseek", now=1010.0)
    assert st_q.in_cooldown is True
    assert st_q.skip_fast is True

    # 连续 4 次 deadline 进入 fast-fail
    for i in range(4):
        store.record_failure("gdelt", ReasonCode.DEADLINE, now=1000.0 + i)
    st_g = store.check("gdelt", now=1010.0)
    assert st_g.in_cooldown is True
    assert st_g.skip_fast is True

    calls = {"deepseek": 0, "gdelt": 0}

    class _SpyEng:
        def __init__(self, eid: str, dom: str):
            self.id = eid
            self.domain = dom
            self.tier = "L0"
            self.deadline_sec = 5.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            calls[self.id] += 1
            return EngineEnvelope(engine=self.id, domain=self.domain, results=[])

    # 使用当前时间戳记录故障供 FederatedPool 实时读取
    live_store = HealthStore(tmp_path / "live_health.json")
    live_store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED)
    for _ in range(4):
        live_store.record_failure("gdelt", ReasonCode.DEADLINE)

    pool = FederatedPool(engines=[_SpyEng("deepseek", "web"), _SpyEng("gdelt", "news")], health_store=live_store, cache_ttl_sec=0.0)
    resp = await pool.search("hello", domains=["web", "news"], limit=2)
    await pool.aclose()

    assert calls["deepseek"] == 0
    assert calls["gdelt"] == 0
    assert resp.coverage.reason_codes.get("deepseek") == "quota-exhausted"
    assert resp.coverage.reason_codes.get("gdelt") == "deadline"

    # reset 后恢复可调用
    live_store.reset("deepseek")
    assert live_store.check("deepseek").in_cooldown is False
    assert live_store.check("deepseek").skip_fast is False


@pytest.mark.asyncio
async def test_fetcher_strips_svg_title_prefers_article_supports_github_issue_llms_txt_and_cache(tmp_path):
    """验证修复 <svg><title> 粘连、优先 <article> 去导航、GitHub Issue API、/llms.txt 与磁盘缓存。"""
    hits: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        hits.append(u)
        if u == "https://docs.example.org/page":
            html = """
            <html>
              <head><title>Web server - Glossary | MDN</title></head>
              <body>
                <header>
                  <svg><title>MDN</title></svg>
                  <svg><title>MDN</title></svg>
                  <svg><title>Mozilla</title></svg>
                  <nav>Sign in | Menu | Notifications</nav>
                </header>
                <main>
                  <div>Repository navigation Fork 6 Star 24</div>
                  <article>
                    <h1>Clean Article Heading</h1>
                    <p>This is the primary article body with concrete technical facts and benchmark metrics.</p>
                    <p>Second paragraph ensures article length exceeds the minimum threshold for primary selection.</p>
                  </article>
                </main>
              </body>
            </html>
            """
            return httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"})
        if u == "https://api.github.com/repos/org/demo/issues/42":
            return httpx.Response(
                200,
                json={
                    "title": "Support Fast-Fail Circuit Breaker",
                    "state": "open",
                    "labels": [{"name": "enhancement"}],
                    "body": "Skip quota-exhausted engines immediately during cooldown.",
                },
                headers={"content-type": "application/json"},
            )
        if u == "https://agentdocs.example.org/llms.txt":
            return httpx.Response(
                200,
                text="# AgentDocs\n\n> Concise documentation index for AI agents.\n\n- /docs/search\n",
                headers={"content-type": "text/plain; charset=utf-8"},
            )
        return httpx.Response(404, text="not found")

    cache = DiskCacheStore(tmp_path / "fetch_cache.db", ttl_sec=300.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        r1 = await safe_fetch_url("https://docs.example.org/page", client=cli, resolve_host=False, disk_cache=cache)
        assert r1.ok is True
        assert r1.title == "Web server - Glossary | MDN"
        assert "MDNMDN" not in r1.title
        assert "Clean Article Heading" in r1.content
        assert "Repository navigation" not in r1.content
        assert "Sign in" not in r1.content
        assert r1.cached is False

        # 二次抓取命中磁盘缓存，不再发 HTTP
        hits_before = len(hits)
        r1_cached = await safe_fetch_url("https://docs.example.org/page", client=cli, resolve_host=False, disk_cache=cache)
        assert r1_cached.ok is True
        assert r1_cached.cached is True
        assert len(hits) == hits_before

        # GitHub Issue 走官方 API
        r_issue = await safe_fetch_url("https://github.com/org/demo/issues/42", client=cli, resolve_host=False)
        assert r_issue.ok is True
        assert r_issue.source_via == "github-issue-api"
        assert "Fast-Fail Circuit Breaker" in r_issue.content

        # /llms.txt 优先探测
        r_llms = await safe_fetch_url(
            "https://agentdocs.example.org/",
            client=cli,
            resolve_host=False,
            prefer_llms_txt=True,
        )
        assert r_llms.ok is True
        assert r_llms.source_via == "llms-txt"
        assert "Concise documentation index" in r_llms.content
