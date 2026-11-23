"""Phase 2C 红灯测试：SQLite 引擎效费比遥测面板（doctor --stats）与 IR 黄金检索质量基准。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, EngineFailure, ReasonCode, SearchItem
from omni_search.health import HealthStore
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import FederatedPool


@pytest.mark.asyncio
async def test_telemetry_store_records_p95_top5_and_recommendations(tmp_path):
    """TelemetryStore 应准确记录耗时 P50/P95、Top-5 贡献与慢源/低成功率调优建议，并与 FederatedPool 集成。"""
    from omni_search.telemetry import TelemetryStore

    tdb = tmp_path / "telemetry.db"
    store = TelemetryStore(tdb)

    class _GoodEng:
        id = "good_src"
        domain = "web"
        tier = "L0"
        deadline_sec = 5.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=120,
                results=[
                    SearchItem(
                        id="1",
                        domain="web",
                        source=self.id,
                        title="SQLite Hybrid Search Guide",
                        url="https://good.example.org/sqlite",
                        snippet="Comprehensive SQLite FTS5 hybrid search guide.",
                    )
                ],
            )

    class _BadSlowEng:
        id = "slow_flaky"
        domain = "web"
        tier = "L0"
        deadline_sec = 5.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=7500,
                failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason="timeout"),
            )

    pool = FederatedPool(
        engines=[_GoodEng(), _BadSlowEng()],
        health_store=HealthStore(tmp_path / "h.json"),
        cache_ttl_sec=0.0,
        telemetry=store,
    )
    for i in range(3):
        await pool.search(f"sqlite hybrid query {i}", domains=["web"], limit=3)
    await pool.aclose()

    summary = store.summarize(window_days=7)
    assert summary["total_events"] >= 6
    by_eng = {e["engine"]: e for e in summary["engines"]}
    assert by_eng["good_src"]["success_rate"] == 1.0
    assert by_eng["good_src"]["top5_contribution_total"] >= 3
    assert by_eng["slow_flaky"]["success_rate"] == 0.0
    assert by_eng["slow_flaky"]["p95_ms"] >= 7000
    assert any("slow_flaky" in rec for rec in summary["recommendations"])


@pytest.mark.asyncio
async def test_doctor_stats_mcp_surface(tmp_path):
    """OmniMCPServer.doctor(stats=True) 应返回遥测统计结构。"""
    srv = OmniMCPServer(health_path=tmp_path / "h.json")
    try:
        report = await srv.doctor(probe=False, stats=True, stats_days=7)
        assert "telemetry" in report
        assert report["telemetry"]["window_days"] == 7
        assert isinstance(report["telemetry"]["engines"], list)
        assert isinstance(report["telemetry"]["recommendations"], list)
    finally:
        await srv.aclose()


@pytest.mark.asyncio
async def test_ir_benchmark_offline_golden_suite_meets_thresholds():
    """离线 IR 黄金评测集应可复现计算 MRR / Hit@3 / Noise@3 / 压缩比，并全部达到质量门槛。"""
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from evals.ir_benchmark import run_offline_ir_benchmark

    res = await run_offline_ir_benchmark()
    metrics = res["metrics"]
    assert res["case_count"] >= 12
    assert metrics["mrr"] >= 0.85
    assert metrics["hit_at_3"] >= 0.90
    assert metrics["noise_at_3"] <= 0.10
    assert metrics["concise_saving_ratio"] >= 0.30
    assert res["pass"] is True
