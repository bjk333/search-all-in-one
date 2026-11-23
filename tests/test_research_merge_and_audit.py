import json
import pytest

from omni_search.contracts import (
    CompletenessReport,
    CoverageReport,
    LedgerSummary,
    SearchItem,
    SearchResponse,
)
from omni_search.fetcher import FetchResult
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool, normalize_url_key
from omni_search.research import (
    audit_freshness_and_conflicts,
    export_research_workorders,
    extract_pub_year,
    extract_relevant_quotes,
    merge_lane_results,
    render_research_markdown,
    run_research_rounds,
)


def _make_resp(query: str, domains: list[str], items: list[SearchItem]) -> dict:
    resp = SearchResponse(
        version="0.1.0",
        query=query,
        domains=domains,
        results=items,
        coverage=CoverageReport(
            requested=["wiki"],
            executed=["wiki"],
            verdict="full",
        ),
        completeness=CompletenessReport(returned_count=len(items), total_candidates=len(items), per_source_limit=5),
        ledger=LedgerSummary(run_id="test123", elapsed_ms=10, budget_ms=5000, requests_attempted=1, requests_succeeded=1),
    )
    return resp.model_dump()


def test_extract_pub_year_and_freshness_conflict_audit():
    assert extract_pub_year("2026-11-20T00:00:00Z", "https://example.com") == 2026
    assert extract_pub_year(None, "http://arxiv.org/abs/2104.10450v1") == 2021
    assert extract_pub_year(None, "https://example.com/no-date") is None

    cands = [
        SearchItem(
            id="1",
            domain="academic",
            source="arxiv",
            title="Old Search Architecture",
            url="http://arxiv.org/abs/2004.01234v1",
            extra={"matched_sources": ["arxiv"], "source_years": {"arxiv": 2020}},
        ),
        SearchItem(
            id="2",
            domain="web",
            source="searxng",
            title="Old Search Architecture",
            url="https://blog.example.com/old-search",
            published_at="2026-05-01",
            extra={"matched_sources": ["searxng"], "source_years": {"searxng": 2026, "archive_org": 2020}},
        ),
        SearchItem(
            id="3",
            domain="academic",
            source="openalex",
            title="Legacy Metasearch",
            url="https://openalex.org/W1",
            published_at="2018-01-01",
        ),
    ]

    audit = audit_freshness_and_conflicts(cands, current_year=2026)
    assert cands[0].extra["pub_year"] == 2020
    assert cands[0].extra["freshness"] == "stale"
    assert cands[1].extra["freshness"] == "recent"
    assert audit.stale_count == 2
    assert audit.recent_count == 1
    assert len(audit.date_discrepancies) >= 1
    assert len(audit.freshness_warnings) >= 1


@pytest.mark.asyncio
async def test_export_workorders_and_strict_merge_rejects_bad_plan_and_conflicts(tmp_path):
    out_dir = tmp_path / "rr"
    plan = export_research_workorders(
        out_dir=out_dir,
        brief="universal search engine",
        mode="factcheck",
        extra_terms=["mcp"],
    )
    assert (out_dir / "plan.json").exists()
    for ln in plan.lanes:
        assert (out_dir / "agents" / f"{ln.name}.md").exists()

    results_dir = out_dir / "agents"
    ln0 = plan.lanes[0]  # authoritative
    ln1 = plan.lanes[1]  # encyclopedia
    # ln2 (news_verify) 故意留空，测 missing_lanes

    item_ok = SearchItem(
        id="a1",
        domain="academic",
        source="arxiv",
        title="Authoritative MCP Spec",
        url="https://arxiv.org/abs/2605.12345",
        published_at="2026-05-10",
    )
    valid_payload = {
        "plan_id": plan.plan_id,
        "lane": ln0.name,
        "query": ln0.query,
        "domains": ln0.domains,
        "response": _make_resp(ln0.query, ln0.domains, [item_ok]),
    }
    # 1. 合法文件 + 重复相同副本（应去重只计一次）
    (results_dir / "agent1.result.json").write_text(json.dumps(valid_payload), encoding="utf-8")
    (results_dir / "agent1_dup.result.json").write_text(json.dumps(valid_payload), encoding="utf-8")

    # 2. plan_id 不匹配 -> 拒收
    bad_plan = dict(valid_payload, plan_id="deadbeef0000")
    (results_dir / "wrong_plan.result.json").write_text(json.dumps(bad_plan), encoding="utf-8")

    # 3. 计划外 lane -> 拒收
    bad_lane = dict(valid_payload, lane="unknown_lane")
    (results_dir / "wrong_lane.result.json").write_text(json.dumps(bad_lane), encoding="utf-8")

    # 4. 同 lane 冲突的两份不同结果 -> 全部排除
    conflict_a = {
        "plan_id": plan.plan_id,
        "lane": ln1.name,
        "query": ln1.query,
        "domains": ln1.domains,
        "response": _make_resp(
            ln1.query,
            ln1.domains,
            [SearchItem(id="c1", domain="knowledge", source="wikipedia", title="Wiki A", url="https://en.wikipedia.org/wiki/A")],
        ),
    }
    conflict_b = {
        "plan_id": plan.plan_id,
        "lane": ln1.name,
        "query": ln1.query,
        "domains": ln1.domains,
        "response": _make_resp(
            ln1.query,
            ln1.domains,
            [SearchItem(id="c2", domain="knowledge", source="wikipedia", title="Wiki B", url="https://en.wikipedia.org/wiki/B")],
        ),
    }
    (results_dir / "enc_a.result.json").write_text(json.dumps(conflict_a), encoding="utf-8")
    (results_dir / "enc_b.result.json").write_text(json.dumps(conflict_b), encoding="utf-8")

    report = await merge_lane_results(work_dir=out_dir, verify_top_k=0, render_markdown=True)

    assert report.merge_audit is not None
    assert ln0.name in report.merge_audit.accepted_lanes
    assert len(report.merge_audit.deduplicated) == 1
    assert len(report.merge_audit.rejected) == 2
    assert ln1.name in report.merge_audit.conflicts
    assert plan.lanes[2].name in report.merge_audit.missing_lanes

    # 冲突 lane 的两条结果均被排除，只保留合法 lane 的结果
    urls = [c.url for c in report.candidates]
    assert "https://arxiv.org/abs/2605.12345" in urls
    assert "https://en.wikipedia.org/wiki/A" not in urls
    assert "https://en.wikipedia.org/wiki/B" not in urls

    md = report.markdown or render_research_markdown(report)
    assert "时效与冲突审计" in md
    assert "多 Agent 合单归属审计" in md


def test_arxiv_doi_url_normalization_and_snippet_date_extraction():
    k1 = normalize_url_key("https://doi.org/10.48550/arxiv.2506.11763")
    k2 = normalize_url_key("http://arxiv.org/abs/2506.11763v2")
    k3 = normalize_url_key("https://arxiv.org/html/2506.11763v1")
    assert k1 == k2 == k3 == "arxiv:2506.11763"

    assert extract_pub_year(None, "https://blog.example.com/post", snippet="Aug 13, 2026 · Review of GPT Researcher") == 2026
    assert extract_pub_year(None, "https://blog.example.com/post", snippet="Feb 4, 2025 ... DeepResearch overview") == 2025


def test_extract_relevant_quotes_skips_nav_noise_and_picks_query_matched_sentences():
    raw = (
        "Repository navigation Notifications You must be signed in to change settings. "
        "Sign up Login Home About Pricing. "
        "WebWeaver introduces dynamic outlines and an evidence memory bank for citation grounding in deep research agents. "
        "Click here for cookies policy."
    )
    quotes = extract_relevant_quotes(raw, query="deep research citation grounding", max_quotes=2)
    assert len(quotes) >= 1
    assert "WebWeaver" in quotes[0]
    assert "signed in" not in quotes[0]


@pytest.mark.asyncio
async def test_research_folds_clone_repos_and_fallbacks_on_verify_failure(tmp_path):
    class _Eng:
        def __init__(self, eid, domain, items):
            self.id = eid
            self.domain = domain
            self.tier = "L0"
            self.deadline_sec = 5.0
            self._items = items

        async def search(self, query, limit, client):
            from omni_search.contracts import EngineEnvelope

            return EngineEnvelope(engine=self.id, domain=self.domain, results=list(self._items))

    clones = [
        SearchItem(id=f"g{i}", domain="code", source="mcps_repo:Gitee", title=f"user{i}/academic-research-skills", url=f"https://gitee.com/user{i}/academic-research-skills", popularity=10 - i)
        for i in range(5)
    ]
    others = [
        SearchItem(id="d1", domain="academic", source="crossref", title="Blocked Paper", url="https://paywall.example.com/p1"),
        SearchItem(id="a1", domain="academic", source="arxiv", title="Open Paper", url="https://arxiv.org/abs/2601.12346"),
        SearchItem(id="w1", domain="web", source="searxng", title="Open Benchmark", url="https://bench.example.org/deep"),
    ]
    pool = FederatedPool(
        engines=[_Eng("mcps_repo", "code", clones), _Eng("acad", "academic", others[:2]), _Eng("web", "web", [others[2]])],
        health_store=HealthStore(tmp_path / "h.json"),
    )

    async def fake_fetch(url: str, max_chars: int = 2000, timeout_sec: float = 10.0) -> FetchResult:
        if "paywall.example.com" in url:
            return FetchResult(ok=False, url=url, status_code=403, reason_code="http-error", reason="HTTP 403")
        return FetchResult(
            ok=True,
            url=url,
            status_code=200,
            title="Verified Doc",
            content="Header nav menu. This deep research benchmark evaluates citation grounding accuracy thoroughly.",
            untrusted_content=True,
        )

    rep = await run_research_rounds(
        pool=pool,
        brief="deep research benchmark citation grounding",
        domains=["code", "academic", "web"],
        limit=5,
        verify_top_k=2,
        fetcher=fake_fetch,
    )

    # 5 个同名克隆仓库应折叠为 1 条主候选并记录 4 个 duplicate_variants
    skill_cands = [c for c in rep.candidates if "academic-research-skills" in c.title]
    assert len(skill_cands) == 1
    assert skill_cands[0].extra.get("duplicate_variants") == 4

    # 虽遇到 1 次 403，应自动顺延补抓，最终仍有 2 条 verified=True 且包含 relevant_quotes
    ok_vers = [v for v in rep.verified_evidence if v.verified]
    assert len(ok_vers) == 2
    assert any(v.relevant_quotes for v in ok_vers)
