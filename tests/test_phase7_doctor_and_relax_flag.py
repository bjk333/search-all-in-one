"""阶段 7B/7C 红灯测试：omni doctor 健康汇总与可执行修复建议 + omni_search(autoRelax=False)。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, ReasonCode, RequestStats
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import FederatedPool


class _StrictFlagEngine:
    id = "strict_flag"
    domain = "code"
    tier = "L0"
    deadline_sec = 5.0

    def __init__(self) -> None:
        self.calls = 0

    def is_ready(self) -> tuple[bool, str]:
        return True, "ok"

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        self.calls += 1
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=[],
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
        )


@pytest.mark.asyncio
async def test_doctor_emits_health_summary_half_open_and_actionable_recommendations(tmp_path) -> None:
    srv = OmniMCPServer(health_path=tmp_path / "h.json", env={})
    try:
        # 制造一个持久配额耗尽的引擎和一个冷却到期的半开引擎
        srv.health_store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=10**12)
        for i in range(5):
            srv.health_store.record_failure("gdelt", ReasonCode.DEADLINE, now=1000.0 + i)

        doc = await srv.doctor(probe=False)
        assert "health_summary" in doc
        hs = doc["health_summary"]
        assert hs["cooldown_count"] >= 1
        assert hs["half_open_count"] >= 1
        assert "recommendations" in doc
        recs = doc["recommendations"]
        assert any("deepseek" in r for r in recs)
        assert any("gdelt" in r or "reset-health" in r for r in recs)
        gdelt_row = next(e for e in doc["engines"] if e["id"] == "gdelt")
        assert gdelt_row["half_open"] is True
    finally:
        await srv.aclose()


@pytest.mark.asyncio
async def test_mcp_omni_search_respects_auto_relax_false(tmp_path) -> None:
    srv = OmniMCPServer(health_path=tmp_path / "h2.json", env={})
    eng = _StrictFlagEngine()
    srv.pool = FederatedPool(engines=[eng], cache_ttl_sec=0.0)
    try:
        await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "omni_search",
                    "arguments": {
                        "query": "how to build MCP server search fetch extract crawl 2026",
                        "domains": ["code"],
                        "autoRelax": False,
                    },
                },
            }
        )
        assert eng.calls == 1
    finally:
        await srv.aclose()
