"""Phase 3D 红灯测试：MCP Streamable HTTP 本机回环常驻服务（JSON + SSE 进度流 + Bearer 鉴权 + Origin 防护）。"""

from __future__ import annotations

import json

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.health import HealthStore
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import FederatedPool


class _QuickMockEng:
    id = "arxiv"
    domain = "academic"
    tier = "L0"
    deadline_sec = 3.0

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=[
                SearchItem(
                    id="p1",
                    domain="academic",
                    source="arxiv",
                    title="Streamable HTTP MCP Paper",
                    url="https://arxiv.org/abs/2610.00001",
                    snippet="MCP Streamable HTTP benchmark.",
                )
            ],
        )


@pytest.mark.asyncio
async def test_mcp_http_health_initialize_tools_and_bearer_auth(tmp_path):
    """验证 /health、Bearer 401、Origin 403、initialize/202 通知/tools/resources/omni_doctor 全链路。"""
    from omni_search.mcp_http import McpHttpServer

    mcp = OmniMCPServer(health_path=tmp_path / "h.json")
    http_srv = McpHttpServer(mcp, host="127.0.0.1", port=0, auth_token="secret-xyz")
    await http_srv.start()
    base = f"http://127.0.0.1:{http_srv.port}"
    try:
        async with httpx.AsyncClient() as cli:
            rh = await cli.get(f"{base}/health")
            assert rh.status_code == 200
            assert rh.json()["server"] == "search-all-in-one"
            assert rh.json()["engines"] == 50

            # 无 Token -> 401
            r_unauth = await cli.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            assert r_unauth.status_code == 401

            # 恶意 Origin -> 403（DNS Rebinding 防护）
            r_evil = await cli.post(
                f"{base}/mcp",
                headers={"Authorization": "Bearer secret-xyz", "Origin": "https://evil.example.com"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            )
            assert r_evil.status_code == 403

            auth_hdr = {"Authorization": "Bearer secret-xyz"}
            r_init = await cli.post(
                f"{base}/mcp",
                headers=auth_hdr,
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            )
            assert r_init.status_code == 200
            assert r_init.headers.get("mcp-session-id")
            assert r_init.json()["result"]["serverInfo"]["name"] == "search-all-in-one"

            # 通知返回 202 Accepted
            r_notif = await cli.post(
                f"{base}/mcp",
                headers=auth_hdr,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert r_notif.status_code == 202

            # tools/list + omni_doctor
            r_tools = await cli.post(
                f"{base}/mcp",
                headers=auth_hdr,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert len(r_tools.json()["result"]["tools"]) == 8

            r_doc = await cli.post(
                f"{base}/mcp",
                headers=auth_hdr,
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "omni_doctor", "arguments": {"probe": False}}},
            )
            assert r_doc.status_code == 200
            assert r_doc.json()["result"]["structuredContent"]["engine_count"] == 50

            # DELETE /mcp -> 204
            r_del = await cli.delete(f"{base}/mcp", headers=auth_hdr)
            assert r_del.status_code == 204
    finally:
        await http_srv.stop()
        await mcp.aclose()


@pytest.mark.asyncio
async def test_mcp_http_sse_streaming_progress_for_research(tmp_path):
    """Accept: text/event-stream 调用 omni_research 时应通过 SSE 推送 progress 事件与最终 JSON-RPC 响应。"""
    from omni_search.mcp_http import McpHttpServer

    mcp = OmniMCPServer(health_path=tmp_path / "h.json")
    mcp.pool = FederatedPool(engines=[_QuickMockEng()], health_store=HealthStore(tmp_path / "h2.json"))
    http_srv = McpHttpServer(mcp, host="127.0.0.1", port=0)
    await http_srv.start()
    base = f"http://127.0.0.1:{http_srv.port}"
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.post(
                f"{base}/mcp",
                headers={"Accept": "text/event-stream, application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {
                        "name": "omni_research",
                        "arguments": {"brief": "Streamable HTTP MCP", "domains": ["academic"], "verifyTopK": 0},
                        "_meta": {"progressToken": "sse-tok-1"},
                    },
                },
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "").lower()
            events = [
                json.loads(line[5:].strip())
                for line in r.text.splitlines()
                if line.startswith("data:")
            ]
            prog_events = [e for e in events if e.get("method") == "notifications/progress"]
            final_events = [e for e in events if e.get("id") == 9]
            assert len(prog_events) >= 2
            assert len(final_events) == 1
            assert final_events[0]["result"]["isError"] is False
    finally:
        await http_srv.stop()
        await mcp.aclose()


def test_mcp_http_refuses_non_loopback_without_token(tmp_path):
    """未配置鉴权 Token 时，严禁监听非回环地址（如 0.0.0.0）。"""
    from omni_search.mcp_http import McpHttpServer

    mcp = OmniMCPServer(health_path=tmp_path / "h.json")
    with pytest.raises(ValueError, match="loopback|token"):
        McpHttpServer(mcp, host="0.0.0.0", port=0, auth_token="")
