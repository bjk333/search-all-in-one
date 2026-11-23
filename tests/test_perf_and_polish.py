import json
import pathlib
import sqlite3

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.declarative import DeclarativeEngine, load_declarative_engines
from omni_search.fetcher import safe_fetch_url
from omni_search.health import HealthStore
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.pool import FederatedPool
from omni_search.research import classify_evidence_tier


@pytest.mark.asyncio
async def test_github_issues_yaml_includes_is_issue_qualifier():
    """github_issues 必须自动附加 is:issue，避免带 GITHUB_TOKEN 时触发 GitHub 422。"""
    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = {s.id: s for s in load_declarative_engines(engines_dir)}
    seen_q = ""

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal seen_q
        seen_q = req.url.params.get("q", "")
        return httpx.Response(200, json={"items": [{"id": 1, "title": "FTS5 bug", "html_url": "https://github.com/o/r/issues/1", "body": "details", "comments": 3}]})

    eng = DeclarativeEngine(specs["github_issues"], env={"GITHUB_TOKEN": "ghp_dummy"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        env = await eng.search("sqlite fts5", limit=2, client=cli)
    assert env.failure is None
    assert "is:issue" in seen_q


def test_dockerhub_is_t2_not_t1():
    """DockerHub 容器镜像源应归为 T2，避免在竞品调研中碾压 GitHub 源码仓与 arXiv 论文。"""
    assert classify_evidence_tier("https://hub.docker.com/r/opensearchproject/opensearch", source="dockerhub") == "T2"


@pytest.mark.asyncio
async def test_fetcher_unescapes_mixed_case_html_entities():
    """fetcher 应将残留的大小写混合 HTML 实体（如 &Amp;）干净解码。"""
    html = "<html><head><title>Tech &amp;amp; Science</title></head><body><main><p>Journal of Engineering &Amp; Science in 2026.</p></main></body></html>"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=html))
    ) as cli:
        res = await safe_fetch_url("https://docs.example.com/ent", client=cli, resolve_host=False)
    assert res.ok is True
    assert "&Amp;" not in res.content
    assert "Engineering & Science" in res.content


@pytest.mark.asyncio
async def test_local_index_uses_wal_mode_and_bm25f_title_boost(tmp_path):
    """本地 SQLite 索引应启用 WAL 模式，且 BM25F 字段加权使文件名/标题命中排在纯正文偶发碰词之前。"""
    ws = tmp_path / "bm25f"
    ws.mkdir()
    (ws / "raft_consensus.md").write_text("# Notes\nBrief design memo about distributed logs.", encoding="utf-8")
    (ws / "random_diary.md").write_text(
        "# Random Diary\n" + ("Today I read a little bit about raft_consensus once. " * 2),
        encoding="utf-8",
    )
    db_path = tmp_path / "wal.db"
    index_paths([ws], db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    finally:
        conn.close()
    assert str(mode).lower() == "wal"

    eng = LocalIndexEngine(db_path=db_path)
    env = await eng.search("raft_consensus", limit=5)
    assert len(env.results) == 2
    assert env.results[0].title == "raft_consensus.md"


@pytest.mark.asyncio
async def test_disk_cache_persists_across_pool_instances_and_skips_local(tmp_path):
    """跨进程 SQLite 磁盘缓存应对公网结果跨池实例命中（cached=True），且绝不缓存 local 域。"""
    from omni_search.cache import DiskCacheStore

    calls = {"web": 0, "local": 0}

    class _CountingEng:
        def __init__(self, eid: str, domain: str):
            self.id = eid
            self.domain = domain
            self.tier = "L0"
            self.deadline_sec = 3.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            calls[self.domain] += 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                results=[SearchItem(id=f"{self.id}-1", domain=self.domain, source=self.id, title=f"Hit {calls[self.domain]}", url=f"https://{self.id}.test/1", snippet="ok")],
            )

    cache_db = tmp_path / "cache.db"
    store1 = DiskCacheStore(cache_db, ttl_sec=300.0)
    pool1 = FederatedPool(engines=[_CountingEng("w1", "web"), _CountingEng("l1", "local")], health_store=HealthStore(tmp_path / "h.json"), disk_cache=store1)
    r1 = await pool1.search("sqlite", domains=["web"], limit=3)
    assert r1.cached is False
    assert calls["web"] == 1
    await pool1.aclose()

    # 新建第二个独立池实例（模拟新 CLI 进程）：公网查询应直接命中磁盘缓存，不再调引擎
    store2 = DiskCacheStore(cache_db, ttl_sec=300.0)
    pool2 = FederatedPool(engines=[_CountingEng("w1", "web"), _CountingEng("l1", "local")], health_store=HealthStore(tmp_path / "h.json"), disk_cache=store2)
    r2 = await pool2.search("sqlite", domains=["web"], limit=3)
    assert r2.cached is True
    assert calls["web"] == 1

    # local 域绝不进磁盘缓存，跨实例仍应实时查询
    await pool2.search("sqlite", domains=["local"], limit=3)
    await pool2.aclose()
    pool3 = FederatedPool(engines=[_CountingEng("w1", "web"), _CountingEng("l1", "local")], health_store=HealthStore(tmp_path / "h.json"), disk_cache=DiskCacheStore(cache_db, ttl_sec=300.0))
    await pool3.search("sqlite", domains=["local"], limit=3)
    await pool3.aclose()
    assert calls["local"] == 2


@pytest.mark.asyncio
async def test_pool_reuses_persistent_http_client_and_concise_view_shrinks_payload(tmp_path):
    """FederatedPool 应复用同一个持久 httpx.AsyncClient，且 concise 精简模式显著压缩返回体积。"""
    from omni_search.contracts import compact_search_response

    seen_client_ids: list[int] = []

    class _ClientSpy:
        id = "spy"
        domain = "web"
        tier = "L0"
        deadline_sec = 3.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            seen_client_ids.append(id(client))
            return EngineEnvelope(
                engine="spy",
                domain="web",
                results=[
                    SearchItem(
                        id="s1",
                        domain="web",
                        source="spy",
                        title="Long Article",
                        url="https://example.com/long",
                        snippet="X" * 480,
                        extra={"huge_meta": "Y" * 400, "matched_sources": ["spy"]},
                    )
                ],
            )

    pool = FederatedPool(engines=[_ClientSpy()], health_store=HealthStore(tmp_path / "h.json"), cache_ttl_sec=0.0)
    try:
        r1 = await pool.search("q1", domains=["web"], limit=3)
        await pool.search("q2", domains=["web"], limit=3)
    finally:
        await pool.aclose()

    assert len(seen_client_ids) == 2
    assert seen_client_ids[0] == seen_client_ids[1]

    full_bytes = len(json.dumps(r1.model_dump(), ensure_ascii=False))
    concise_obj = compact_search_response(r1)
    concise_bytes = len(json.dumps(concise_obj, ensure_ascii=False))
    assert concise_bytes < full_bytes * 0.7
    assert concise_obj["coverage"]["verdict"] == "full"
    assert len(concise_obj["results"][0]["snippet"]) <= 165
