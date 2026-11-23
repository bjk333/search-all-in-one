"""Phase 2D 红灯测试：MCP resources/prompts/progress 通知 + omni monitor 持续增量快照监控。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.fetcher import FetchResult
from omni_search.health import HealthStore
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import FederatedPool


class _StaticEng:
    def __init__(self, eid: str, dom: str, items_ref: list[SearchItem]):
        self.id = eid
        self.domain = dom
        self.tier = "L0"
        self.deadline_sec = 5.0
        self._items_ref = items_ref

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        return EngineEnvelope(engine=self.id, domain=self.domain, results=list(self._items_ref))


@pytest.mark.asyncio
async def test_mcp_capabilities_resources_prompts_and_progress_notifications(tmp_path):
    """MCP Server 应支持 resources/prompts 与 omni_research 的 notifications/progress 推送。"""
    notifications: list[dict] = []
    srv = OmniMCPServer(health_path=tmp_path / "h.json", emit_notification=notifications.append)
    # 替换为离线 mock 引擎加速且确定性运行
    sample = [
        SearchItem(
            id="p1",
            domain="academic",
            source="arxiv",
            title="MCP Progress Protocol Paper",
            url="https://arxiv.org/abs/2609.12345",
            snippet="Progress notifications for long-running MCP tools.",
            published_at="2026-09-01",
        )
    ]
    srv.pool = FederatedPool(engines=[_StaticEng("arxiv", "academic", sample)], health_store=HealthStore(tmp_path / "h2.json"))

    try:
        init = await srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        caps = init["result"]["capabilities"]
        assert "resources" in caps and "prompts" in caps

        rlist = await srv.handle_request({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        uris = {r["uri"] for r in rlist["result"]["resources"]}
        assert "omni://research/last-report" in uris
        assert "omni://doctor/telemetry" in uris
        assert "omni://local/index/summary" in uris

        plist = await srv.handle_request({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})
        pnames = {p["name"] for p in plist["result"]["prompts"]}
        assert {"feasibility_research", "benchmark_comparison", "security_cve_audit"}.issubset(pnames)

        pget = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "prompts/get",
                "params": {"name": "feasibility_research", "arguments": {"topic": "sqlite-vec"}},
            }
        )
        assert "sqlite-vec" in pget["result"]["messages"][0]["content"]["text"]

        # 带 progressToken 调用 omni_research
        res = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "omni_research",
                    "arguments": {"brief": "MCP progress protocol", "domains": ["academic"], "verifyTopK": 0},
                    "_meta": {"progressToken": "tok-42"},
                },
            }
        )
        assert res["result"]["isError"] is False
        assert len(notifications) >= 2
        assert all(n["method"] == "notifications/progress" for n in notifications)
        assert all(n["params"]["progressToken"] == "tok-42" for n in notifications)

        # 读取最近简报资源
        read_res = await srv.handle_request(
            {"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": "omni://research/last-report"}}
        )
        text = read_res["result"]["contents"][0]["text"]
        assert "MCP progress protocol" in text
    finally:
        await srv.aclose()


@pytest.mark.asyncio
async def test_monitor_baseline_and_incremental_diff_detection(tmp_path):
    """omni monitor 应首次建基线，二次发现新增 T1 条目，三次无变化报 has_changes=False。"""
    from omni_search.monitor import run_monitor_check

    items = [
        SearchItem(
            id="a1",
            domain="academic",
            source="arxiv",
            title="Baseline Paper on SQLite Search",
            url="https://arxiv.org/abs/2608.00001",
            snippet="Baseline sqlite search study.",
            published_at="2026-08-01",
        )
    ]

    async def _fake_fetch(url: str, max_chars: int = 3000, timeout_sec: float = 10.0) -> FetchResult:
        return FetchResult(ok=True, url=url, status_code=200, title="Paper", content="SQLite search verified text.", untrusted_content=True)

    pool = FederatedPool(engines=[_StaticEng("arxiv", "academic", items)], health_store=HealthStore(tmp_path / "h.json"), cache_ttl_sec=0.0)
    state_dir = tmp_path / "monitors"
    try:
        # Round 1: 基线
        m1 = await run_monitor_check(pool, "SQLite Search", state_dir=state_dir, domains=["academic"], verify_top_k=1, fetcher=_fake_fetch)
        assert m1.is_baseline is True
        assert m1.has_changes is False

        # Round 2: 新增一篇 T1 论文
        items.append(
            SearchItem(
                id="a2",
                domain="academic",
                source="arxiv",
                title="New 2026 Breakthrough on SQLite Vector Search",
                url="https://arxiv.org/abs/2611.99999",
                snippet="Brand new sqlite vector paper.",
                published_at="2026-11-20",
            )
        )
        m2 = await run_monitor_check(pool, "SQLite Search", state_dir=state_dir, domains=["academic"], verify_top_k=1, fetcher=_fake_fetch)
        assert m2.is_baseline is False
        assert m2.has_changes is True
        assert m2.added_count == 1
        assert len(m2.added_t1) == 1
        assert "2611.99999" in m2.added_t1[0].url

        # Round 3: 无新增变更
        m3 = await run_monitor_check(pool, "SQLite Search", state_dir=state_dir, domains=["academic"], verify_top_k=1, fetcher=_fake_fetch)
        assert m3.is_baseline is False
        assert m3.has_changes is False
        assert m3.added_count == 0
    finally:
        await pool.aclose()
