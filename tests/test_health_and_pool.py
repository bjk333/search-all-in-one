import asyncio
import pytest

from omni_search.contracts import EngineEnvelope, EngineFailure, ReasonCode, SearchItem
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool, fuse_results_rrf
from omni_search.router import route_domains


class DummyEngine:
    def __init__(self, engine_id: str, domain: str, delay: float = 0.0, items=None, failure=None):
        self.id = engine_id
        self.domain = domain
        self.tier = "L0"
        self.deadline_sec = 5.0
        self._delay = delay
        self._items = items or []
        self._failure = failure

    async def search(self, query: str, limit: int, client) -> EngineEnvelope:
        if self._delay:
            await asyncio.sleep(self._delay)
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=list(self._items),
            elapsed_ms=int(self._delay * 1000),
            failure=self._failure,
        )


def test_route_domains_auto_selects_appropriate_domains():
    assert "code" in route_domains("rust async http client github", ["auto"])
    assert "academic" in route_domains("arxiv transformer attention paper", ["auto"])
    assert "news" in route_domains("latest ai news today", ["auto"])
    assert set(route_domains("quantum computing", ["knowledge", "academic"])) == {"knowledge", "academic"}


def test_rrf_deduplicates_normalized_urls_and_boosts_multi_source_hits():
    shared_url = "https://example.com/post/?utm_source=hn"
    same_url = "https://example.com/post"
    e1 = EngineEnvelope(
        engine="src_a",
        domain="web",
        results=[
            SearchItem(id="1", domain="web", source="src_a", title="Shared", url=shared_url),
            SearchItem(id="2", domain="web", source="src_a", title="Only A", url="https://a.example.com/1"),
        ],
    )
    e2 = EngineEnvelope(
        engine="src_b",
        domain="news",
        results=[
            SearchItem(id="3", domain="news", source="src_b", title="Only B", url="https://b.example.com/1"),
            SearchItem(id="4", domain="news", source="src_b", title="Shared", url=same_url),
        ],
    )
    fused = fuse_results_rrf([e1, e2])
    assert len(fused) == 3
    # 同时被两源命中的 Shared 应排第一
    assert fused[0].title == "Shared"
    assert set(fused[0].extra.get("matched_sources", [])) == {"src_a", "src_b"}


def test_health_store_triggers_cooldown_after_failures(tmp_path):
    store = HealthStore(tmp_path / "health.json")
    store.record_failure("codeberg", ReasonCode.RATE_LIMITED, now=1000.0)
    store.record_failure("codeberg", ReasonCode.RATE_LIMITED, now=1001.0)
    state = store.check("codeberg", now=1005.0)
    assert state.in_cooldown is True
    assert state.probe_deadline_sec <= 6.0

    # 成功后重置健康状态
    store.record_success("codeberg", now=1010.0)
    assert store.check("codeberg", now=1011.0).in_cooldown is False


@pytest.mark.asyncio
async def test_pool_enforces_engine_deadline_without_blocking_fast_engines(tmp_path):
    fast = DummyEngine(
        "fast_wiki",
        "knowledge",
        delay=0.02,
        items=[SearchItem(id="w1", domain="knowledge", source="fast_wiki", title="Fast Hit", url="https://wiki.example/fast")],
    )
    slow = DummyEngine("slow_src", "web", delay=2.0)
    slow.deadline_sec = 0.15

    pool = FederatedPool(engines=[fast, slow], health_store=HealthStore(tmp_path / "h.json"))
    resp = await pool.search(query="fast", domains=["knowledge", "web"], limit=5, budget_ms=1000)

    assert resp.coverage.verdict == "degraded"
    assert "fast_wiki" in resp.coverage.executed
    assert "slow_src" in resp.coverage.degraded
    assert resp.coverage.reason_codes["slow_src"] == ReasonCode.DEADLINE.value
    assert len(resp.results) == 1
    assert resp.results[0].title == "Fast Hit"


@pytest.mark.asyncio
async def test_pool_query_cache_hits_on_repeat_and_skips_empty(tmp_path):
    calls = {"n": 0}

    class CountingEngine(DummyEngine):
        async def search(self, query: str, limit: int, client) -> EngineEnvelope:
            calls["n"] += 1
            return await super().search(query, limit, client)

    eng = CountingEngine(
        "wiki",
        "knowledge",
        items=[SearchItem(id="1", domain="knowledge", source="wiki", title="Quantum Computing", url="https://wiki.test/q")],
    )
    pool = FederatedPool(engines=[eng], health_store=HealthStore(tmp_path / "hc.json"), cache_ttl_sec=60.0)

    r1 = await pool.search("quantum", domains=["knowledge"], limit=5)
    r2 = await pool.search("quantum", domains=["knowledge"], limit=5)
    assert calls["n"] == 1
    assert r1.cached is False
    assert r2.cached is True
    assert len(r2.results) == 1

    # 空结果不得缓存
    empty_eng = CountingEngine("empty_src", "news", items=[])
    pool_e = FederatedPool(engines=[empty_eng], health_store=HealthStore(tmp_path / "he.json"), cache_ttl_sec=60.0)
    await pool_e.search("nothing", domains=["news"], limit=5)
    await pool_e.search("nothing", domains=["news"], limit=5)
    assert calls["n"] == 3


def test_rrf_relevance_boost_prefers_title_match_over_unrelated_rank1():
    e1 = EngineEnvelope(
        engine="src_a",
        domain="web",
        results=[SearchItem(id="1", domain="web", source="src_a", title="Completely Unrelated Popular Page", url="https://a.test/1")],
    )
    e2 = EngineEnvelope(
        engine="src_b",
        domain="knowledge",
        results=[SearchItem(id="2", domain="knowledge", source="src_b", title="Metasearch Engine Architecture", url="https://b.test/2")],
    )
    fused = fuse_results_rrf([e1, e2], query="metasearch engine")
    assert fused[0].title == "Metasearch Engine Architecture"
