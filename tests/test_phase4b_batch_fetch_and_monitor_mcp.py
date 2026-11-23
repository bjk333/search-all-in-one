"""阶段 4B 红灯测试：并发批量安全抓取 (safe_fetch_urls_batch) 与 MCP omni_fetch(urls)/omni_monitor。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, RequestStats, SearchItem
from omni_search.fetcher import safe_fetch_urls_batch
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import FederatedPool


class _StubMonitorEngine:
    id = "stub_mon"
    domain = "code"
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
async def test_safe_fetch_urls_batch_isolates_ssrf_and_dedups() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/doc1":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><head><title>Doc One</title></head><body><main><p>FTS5 hybrid search details here.</p></main></body></html>",
            )
        if request.url.path == "/missing":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as cli:
        res = await safe_fetch_urls_batch(
            urls=[
                "https://docs.example.com/doc1",
                "https://docs.example.com/doc1",  # 重复 URL 应自动去重
                "http://127.0.0.1:8080/secret",   # SSRF 应单条隔离拦截
                "https://docs.example.com/missing",
            ],
            query="FTS5 hybrid",
            concurrency=3,
            client=cli,
            resolve_host=False,
        )

    assert res.ok is True
    assert res.untrusted_content is True
    assert res.total == 3  # 去重后 3 条
    assert res.succeeded == 1
    assert res.failed == 2
    by_url = {it.url: it for it in res.items}
    assert by_url["https://docs.example.com/doc1"].ok is True
    assert "FTS5 hybrid" in by_url["https://docs.example.com/doc1"].content
    assert by_url["http://127.0.0.1:8080/secret"].reason_code == "ssrf-blocked"
    assert by_url["https://docs.example.com/missing"].reason_code == "http-error"


@pytest.mark.asyncio
async def test_mcp_omni_fetch_urls_batch_and_omni_monitor(tmp_path) -> None:
    eng = _StubMonitorEngine(
        [
            SearchItem(
                id="r1",
                domain="code",
                source="stub_mon",
                title="Repo Alpha Framework",
                url="https://github.com/org/alpha",
                snippet="alpha framework release v1",
            )
        ]
    )
    srv = OmniMCPServer(env={})
    srv.pool = FederatedPool(engines=[eng], cache_ttl_sec=0)
    try:
        # tools/list 必须包含 omni_monitor
        listed = await srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in listed["result"]["tools"]}
        assert "omni_monitor" in names

        # omni_fetch 传入 urls 列表（全部是本机 SSRF 地址，无需外网即可确定性验证批量路径）
        f_resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "omni_fetch",
                    "arguments": {
                        "urls": ["http://127.0.0.1/a", "http://169.254.169.254/latest/meta-data/"],
                    },
                },
            }
        )
        sc = f_resp["result"]["structuredContent"]
        assert sc["total"] == 2
        assert sc["failed"] == 2
        assert sc["untrusted_content"] is True
        assert all(it["reason_code"] == "ssrf-blocked" for it in sc["items"])

        # omni_monitor 首次建立基线，二次检测新增 T1 变更
        m1 = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "omni_monitor",
                    "arguments": {
                        "topic": "alpha framework",
                        "domains": ["code"],
                        "verifyTopK": 0,
                        "stateDir": str(tmp_path / "mon"),
                        "renderMarkdown": True,
                    },
                },
            }
        )
        sc1 = m1["result"]["structuredContent"]
        assert sc1["is_baseline"] is True
        assert sc1["total_candidates"] >= 1
        assert "主题增量监控报告" in sc1["markdown"]

        eng._items.append(
            SearchItem(
                id="r2",
                domain="code",
                source="stub_mon",
                title="Repo Beta Framework",
                url="https://github.com/org/beta",
                snippet="alpha framework beta plugin v2",
            )
        )
        m2 = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "omni_monitor",
                    "arguments": {
                        "topic": "alpha framework",
                        "domains": ["code"],
                        "verifyTopK": 0,
                        "stateDir": str(tmp_path / "mon"),
                    },
                },
            }
        )
        sc2 = m2["result"]["structuredContent"]
        assert sc2["is_baseline"] is False
        assert sc2["has_changes"] is True
        assert sc2["added_count"] == 1
    finally:
        await srv.aclose()
