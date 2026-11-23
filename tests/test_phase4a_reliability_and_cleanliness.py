"""阶段 4A 红灯测试：半开熔断超时收紧、HTML 实体解码清洗、版本号与 MCP 元数据一致性。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest

import omni_search
from omni_search.contracts import EngineEnvelope, ReasonCode, RequestStats, SearchItem
from omni_search.declarative import _apply_transform, extract_path
from omni_search.health import HealthStore
from omni_search.mcp_server import _TOOLS_SPEC
from omni_search.pool import FederatedPool


class _SlowStubEngine:
    id = "slow_half_open"
    domain = "news"
    tier = "L0"
    deadline_sec = 12.0

    def __init__(self, sleep_sec: float = 0.6, should_succeed: bool = False):
        self.sleep_sec = sleep_sec
        self.should_succeed = should_succeed
        self.calls = 0

    def is_ready(self) -> tuple[bool, str]:
        return True, "ok"

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        self.calls += 1
        await asyncio.sleep(self.sleep_sec)
        if self.should_succeed:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                items=[
                    SearchItem(
                        id="1",
                        domain=self.domain,
                        source=self.id,
                        title="Recovered",
                        url="https://example.com/ok",
                        snippet="ok",
                    )
                ],
                requests=RequestStats(attempted=1, succeeded=1, failed=0),
            )
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            items=[],
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
        )


def test_health_store_half_open_after_cooldown_expires(tmp_path) -> None:
    store = HealthStore(tmp_path / "health.json", probe_deadline_sec=4.0)
    t0 = 1000.0
    # 连续失败 5 次进入 fast-fail 冷却
    for i in range(5):
        store.record_failure("gdelt", ReasonCode.DEADLINE, now=t0 + i)

    st_cd = store.check("gdelt", now=t0 + 10.0)
    assert st_cd.in_cooldown is True
    assert st_cd.skip_fast is True

    # 冷却到期后：必须进入 half_open 状态，仍收紧 probe_deadline_sec，而非放开成 0
    st_ho = store.check("gdelt", now=st_cd.cooldown_until + 1.0)
    assert st_ho.in_cooldown is False
    assert st_ho.skip_fast is False
    assert st_ho.half_open is True
    assert st_ho.probe_deadline_sec == 4.0

    # 探活成功后彻底恢复闭合态
    store.record_success("gdelt", now=st_cd.cooldown_until + 2.0)
    st_ok = store.check("gdelt", now=st_cd.cooldown_until + 3.0)
    assert st_ok.half_open is False
    assert st_ok.in_cooldown is False
    assert st_ok.probe_deadline_sec == 0.0
    assert st_ok.consecutive_failures == 0


@pytest.mark.asyncio
async def test_pool_enforces_short_probe_deadline_in_half_open(tmp_path) -> None:
    store = HealthStore(tmp_path / "health.json", probe_deadline_sec=0.15)
    # 制造历史连续失败，并把 cooldown_until 设为过去（冷却刚到期）
    t_past = time.time() - 3600.0
    for i in range(5):
        store.record_failure("slow_half_open", ReasonCode.DEADLINE, now=t_past + i)

    eng = _SlowStubEngine(sleep_sec=0.8)
    pool = FederatedPool(engines=[eng], health_store=store)
    try:
        t0 = time.perf_counter()
        resp = await pool.search("test", domains=["news"], limit=3, budget_ms=5000, use_cache=False)
        elapsed = time.perf_counter() - t0
    finally:
        await pool.aclose()

    # 半开探测必须按 probe_deadline_sec (0.15s) 截断，绝不能等满 0.8s 或 12s
    assert elapsed < 0.55
    assert eng.calls == 1
    assert resp.coverage.reason_codes.get("slow_half_open") == ReasonCode.DEADLINE
    assert any("半开探测" in w for w in resp.warnings)


def test_strip_html_and_collapse_ws_decode_entities() -> None:
    raw = "<b>Top&nbsp;&nbsp;APIs</b> &amp; &#39;AI&#39; &quot;2026&quot; — MarkTechPost"
    cleaned = _apply_transform(raw, "strip_html")
    assert "&nbsp;" not in cleaned
    assert "&amp;" not in cleaned
    assert "&#39;" not in cleaned
    assert "&quot;" not in cleaned
    assert "\xa0" not in cleaned
    assert cleaned == "Top APIs & 'AI' \"2026\" — MarkTechPost"

    ws = _apply_transform("hello\xa0\xa0  world\n\t!", "collapse_ws")
    assert ws == "hello world !"

    obj: dict[str, Any] = {"desc": "<a>News&nbsp;&nbsp;Source</a>"}
    assert extract_path(obj, "desc | strip_html") == "News Source"


def test_version_and_mcp_metadata_synced() -> None:
    assert omni_search.__version__ == "1.10.0"
    search_spec = next(t for t in _TOOLS_SPEC if t["name"] == "omni_search")
    assert "46 原生引擎" not in search_spec["description"]
