from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import httpx
import pytest

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool
from omni_search.research import (
    ResearchReport,
    VerifiedEvidence,
    render_research_markdown,
)


class _StaticEngine:
    def __init__(self, eid: str, domain: str, items: list[SearchItem]):
        self.id = eid
        self.domain = domain
        self.tier = "L0"
        self.deadline_sec = 3.0
        self._items = items

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        return EngineEnvelope(engine=self.id, domain=self.domain, results=self._items[:limit])


@pytest.mark.asyncio
async def test_http_reranker_real_server_reorders_and_annotates(tmp_path):
    """通过真实本地 HTTP Server 验证 /v1/rerank 协议、鉴权、重排顺序与 extra 透明标注。"""
    from omni_search.reranker import HttpReranker

    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured["path"] = self.path
            captured["auth"] = self.headers.get("Authorization")
            captured["model"] = body.get("model")
            docs = body.get("documents") or []
            # 刻意把包含 "sqlite-vec" 的文档评为最高分 0.96，其余低分
            results = []
            for idx, txt in enumerate(docs):
                score = 0.96 if "sqlite-vec" in str(txt).lower() else 0.12
                results.append({"index": idx, "relevance_score": score})
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            raw = json.dumps({"results": results}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        items = [
            SearchItem(id="1", domain="web", source="a", title="Generic SQLite FAQ", url="https://a.test/1", snippet="General database FAQ"),
            SearchItem(id="2", domain="web", source="a", title="Deep Dive sqlite-vec Hybrid", url="https://a.test/2", snippet="Uses sqlite-vec extension"),
        ]
        reranker = HttpReranker(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="bge-reranker-v2-m3",
            api_key="sk-rerank-test",
        )
        pool = FederatedPool(
            engines=[_StaticEngine("a", "web", items)],
            health_store=HealthStore(tmp_path / "h.json"),
            reranker=reranker,
        )
        resp = await pool.search("sqlite hybrid", domains=["web"], limit=5)

        assert captured["path"] == "/v1/rerank"
        assert captured["auth"] == "Bearer sk-rerank-test"
        assert captured["model"] == "bge-reranker-v2-m3"
        assert resp.results[0].title == "Deep Dive sqlite-vec Hybrid"
        assert resp.results[0].extra["rerank_score"] == pytest.approx(0.96, abs=1e-4)
        assert resp.results[0].extra["reranker"] == "http_rerank:bge-reranker-v2-m3"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_pool_degrades_gracefully_when_reranker_fails(tmp_path):
    """精排服务不可达时必须保留原 RRF 排序并在 warnings 中透明披露，不得导致整次搜索失败。"""

    class BrokenReranker:
        id = "http_rerank:broken"

        async def rerank(self, query: str, items: list[SearchItem], client: httpx.AsyncClient, top_n: int = 20):
            raise RuntimeError("rerank upstream 503")

    items = [
        SearchItem(id="1", domain="web", source="a", title="First RRF Item", url="https://a.test/1", snippet="First"),
        SearchItem(id="2", domain="web", source="a", title="Second RRF Item", url="https://a.test/2", snippet="Second"),
    ]
    pool = FederatedPool(
        engines=[_StaticEngine("a", "web", items)],
        health_store=HealthStore(tmp_path / "h.json"),
        reranker=BrokenReranker(),
    )
    resp = await pool.search("first", domains=["web"], limit=5)
    assert resp.coverage.verdict == "full"
    assert [r.title for r in resp.results] == ["First RRF Item", "Second RRF Item"]
    assert any("rerank failed" in w.lower() for w in resp.warnings)


def test_diff_research_reports_tracks_added_removed_and_verify_regressions():
    """历史快照对比应规范化 URL、识别新增/移除候选、新失效核验链接并渲染到 Markdown 简报。"""
    from omni_search.research import diff_research_reports

    prev = ResearchReport(
        version="0.1.0",
        plan_id="prev0001",
        brief="sqlite hybrid",
        mode="benchmark",
        lanes=[],
        tier_counts={"T1": 2, "T2": 0, "T3": 0},
        candidates=[
            SearchItem(id="p1", domain="academic", source="arxiv", title="scrydb v1", url="http://arxiv.org/abs/2608.24060v1", extra={"evidence_tier": "T1"}),
            SearchItem(id="p2", domain="web", source="searxng", title="Old Blog Post", url="https://old.example.com/post", extra={"evidence_tier": "T1"}),
        ],
        verified_evidence=[
            VerifiedEvidence(url="https://old.example.com/post", title="Old Blog Post", domain="web", source="searxng", evidence_tier="T1", verified=True, status_code=200),
        ],
    )
    curr = ResearchReport(
        version="0.1.0",
        plan_id="curr0002",
        brief="sqlite hybrid",
        mode="benchmark",
        lanes=[],
        tier_counts={"T1": 3, "T2": 0, "T3": 0},
        candidates=[
            # 同 arXiv 论文升级到 v2 -> 应识别为重合保留，而非重复新增
            SearchItem(id="c1", domain="academic", source="arxiv", title="scrydb v2", url="https://arxiv.org/pdf/2608.24060v2", extra={"evidence_tier": "T1"}),
            SearchItem(id="c2", domain="code", source="github_direct", title="new/sqlite-rag", url="https://github.com/new/sqlite-rag", extra={"evidence_tier": "T1"}),
        ],
        verified_evidence=[
            VerifiedEvidence(url="https://old.example.com/post", title="Old Blog Post", domain="web", source="searxng", evidence_tier="T1", verified=False, status_code=404, reason_code="http-error"),
            VerifiedEvidence(url="https://github.com/new/sqlite-rag", title="new/sqlite-rag", domain="code", source="github_direct", evidence_tier="T1", verified=True, status_code=200),
        ],
    )

    diff = diff_research_reports(prev, curr)
    assert diff.prev_plan_id == "prev0001"
    assert diff.retained_count == 1
    assert [x.title for x in diff.added] == ["new/sqlite-rag"]
    assert [x.title for x in diff.removed] == ["Old Blog Post"]
    assert "https://old.example.com/post" in diff.newly_failed_verify
    assert "https://github.com/new/sqlite-rag" in diff.newly_verified
    assert diff.tier_delta["T1"] == 1

    curr.diff = diff
    md = render_research_markdown(curr)
    assert "历史快照变更对比" in md
    assert "new/sqlite-rag" in md
    assert "Old Blog Post" in md
    assert "https://old.example.com/post" in md


@pytest.mark.asyncio
async def test_cli_research_offline_diff_two_snapshots(tmp_path):
    """CLI `omni research --diff prev.json curr.json --md diff.md` 应纯离线输出变更审计与简报。"""
    import argparse
    from omni_search.cli import _run

    prev = ResearchReport(
        version="0.1.0",
        plan_id="p111",
        brief="demo",
        mode="standard",
        lanes=[],
        candidates=[SearchItem(id="1", domain="web", source="a", title="Alpha", url="https://a.test/1", extra={"evidence_tier": "T2"})],
        tier_counts={"T1": 0, "T2": 1, "T3": 0},
    )
    curr = ResearchReport(
        version="0.1.0",
        plan_id="c222",
        brief="demo",
        mode="standard",
        lanes=[],
        candidates=[
            SearchItem(id="1", domain="web", source="a", title="Alpha", url="https://a.test/1", extra={"evidence_tier": "T2"}),
            SearchItem(id="2", domain="code", source="github_direct", title="Beta Repo", url="https://github.com/o/beta", extra={"evidence_tier": "T1"}),
        ],
        tier_counts={"T1": 1, "T2": 1, "T3": 0},
    )
    prev_p = tmp_path / "prev.json"
    curr_p = tmp_path / "curr.json"
    out_p = tmp_path / "diff_out.json"
    md_p = tmp_path / "diff.md"
    prev_p.write_text(prev.model_dump_json(), encoding="utf-8")
    curr_p.write_text(curr.model_dump_json(), encoding="utf-8")

    args = argparse.Namespace(
        cmd="research",
        brief=str(curr_p),
        mode="standard",
        terms=None,
        domains=None,
        limit=4,
        verify_top_k=0,
        budget_ms=10000,
        plan_only=False,
        export_lanes=None,
        merge=None,
        diff=str(prev_p),
        md=str(md_p),
        bib=None,
        include_local=False,
        deepen=False,
        compact=True,
        out=str(out_p),
    )
    rc = await _run(args)
    assert rc == 0
    saved = json.loads(out_p.read_text(encoding="utf-8"))
    assert saved["diff"]["retained_count"] == 1
    assert len(saved["diff"]["added"]) == 1
    assert "Beta Repo" in md_p.read_text(encoding="utf-8")
