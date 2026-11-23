"""omni-search 跨领域回归评测门禁（默认离线确定性评测；加 --live 跑公网端到端评测）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import tempfile
import time
from typing import Any

import httpx

from omni_search.contracts import EngineEnvelope, EngineFailure, ReasonCode, SearchItem, redact_url_secrets, sanitize_text
from omni_search.declarative import DeclarativeEngine, load_declarative_engines
from omni_search.fetcher import safe_fetch_url
from omni_search.health import HealthStore
from omni_search.mcp_server import DEFAULT_ENGINES_DIR, OmniMCPServer
from omni_search.pool import FederatedPool
from omni_search.research import run_research_rounds
from omni_search.router import route_domains


class _StubEngine:
    def __init__(self, eid: str, domain: str, items=None, failure=None, delay: float = 0.0):
        self.id = eid
        self.domain = domain
        self.tier = "L0"
        self.deadline_sec = 0.3
        self._items = items or []
        self._failure = failure
        self._delay = delay

    async def search(self, query: str, limit: int, client) -> EngineEnvelope:
        if self._delay:
            await asyncio.sleep(self._delay)
        return EngineEnvelope(engine=self.id, domain=self.domain, results=list(self._items), failure=self._failure)


async def run_offline_evals() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def record(cid: str, name: str, ok: bool, detail: str = "") -> None:
        cases.append({"id": cid, "name": name, "pass": ok, "detail": detail})

    # E01: 全部声明式 YAML 引擎可加载且不少于 19 个
    specs = load_declarative_engines(DEFAULT_ENGINES_DIR)
    record("E01", "YAML 声明式引擎加载校验", len(specs) >= 19, f"loaded={len(specs)}")

    # E02-E05: 意图路由覆盖（代码 / 学术 / 新闻 / 显式域）
    record("E02", "路由-代码关键词命中 code/packages", {"code", "packages"}.issubset(route_domains("rust tokio github repo", ["auto"])))
    record("E03", "路由-学术关键词命中 academic", "academic" in route_domains("arxiv multimodal retrieval survey", ["auto"]))
    record("E04", "路由-新闻关键词命中 news", "news" in route_domains("latest ai launch news 2026", ["auto"]))
    record("E05", "路由-显式域精准约束", route_domains("anything", ["media", "knowledge"]) == ["media", "knowledge"])

    # E06: 防注入与凭据脱敏
    cleaned = sanitize_text("ok\u200b\u202ehidden")
    redacted = redact_url_secrets("https://x.test/s?api_key=sk123&q=ok")
    record("E06", "安全-控制字符剥离与 URL 脱敏", cleaned == "okhidden" and "sk123" not in redacted)

    # E07: SSRF 拦截（内网/回环/元数据/file 协议）
    ssrf_targets = ["http://127.0.0.1/", "http://localhost:8080/", "http://169.254.169.254/latest/", "file:///etc/hosts"]
    ssrf_res = [await safe_fetch_url(u) for u in ssrf_targets]
    record("E07", "安全-SSRF 全负例阻断", all((not r.ok and r.reason_code == "ssrf-blocked") for r in ssrf_res))

    # E08: 0 命中且源阻塞 => low_confidence_negative=True（绝不伪装不存在）
    with tempfile.TemporaryDirectory() as td:
        pool_blk = FederatedPool(
            engines=[_StubEngine("brave", "web", failure=EngineFailure(reason_code=ReasonCode.MISSING_CREDENTIAL, reason="no key"))],
            health_store=HealthStore(pathlib.Path(td) / "h1.json"),
        )
        r_blk = await pool_blk.search("rare term", domains=["web"])
        record("E08", "契约-全阻塞 0 命中标低置信否定", r_blk.coverage.verdict == "none" and r_blk.low_confidence_negative is True)

        # E09: 单源超时不拖垮快源
        fast = _StubEngine("fast", "web", items=[SearchItem(id="1", domain="web", source="fast", title="Fast", url="https://a.test/1")])
        slow = _StubEngine("slow", "web", delay=1.2)
        pool_dl = FederatedPool(engines=[fast, slow], health_store=HealthStore(pathlib.Path(td) / "h2.json"))
        r_dl = await pool_dl.search("q", domains=["web"], budget_ms=1000)
        record(
            "E09",
            "韧性-单源 Deadline 隔离",
            r_dl.coverage.verdict == "degraded" and r_dl.coverage.reason_codes.get("slow") == "deadline" and len(r_dl.results) == 1,
        )

        # E10: RRF 跨源去重 + 多源交叉上浮
        e_a = _StubEngine(
            "a",
            "web",
            items=[
                SearchItem(id="a1", domain="web", source="a", title="A-only", url="https://a.test/only"),
                SearchItem(id="a2", domain="web", source="a", title="Shared", url="https://shared.test/doc?utm_source=x"),
            ],
        )
        e_b = _StubEngine(
            "b",
            "knowledge",
            items=[
                SearchItem(id="b1", domain="knowledge", source="b", title="Shared", url="https://shared.test/doc/"),
            ],
        )
        pool_rrf = FederatedPool(engines=[e_a, e_b], health_store=HealthStore(pathlib.Path(td) / "h3.json"))
        r_rrf = await pool_rrf.search("doc", domains=["web", "knowledge"])
        record("E10", "融合-RRF 去重与交叉印证置顶", len(r_rrf.results) == 2 and r_rrf.results[0].title == "Shared")

        # E11: 402 余额不足与 min_score 全滤光语义
        spec_map = {s.id: s for s in specs}
        ds = DeclarativeEngine(spec_map["deepseek"], env={"DEEPSEEK_API_KEY": "sk-dummy"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(402))) as cli:
            env_402 = await ds.search("q", 3, cli)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"results": [{"title": "x", "url": "https://x.t", "rerank_score": 0.1}]}))
        ) as cli:
            env_flt = await ds.search("q", 3, cli)
        record(
            "E11",
            "契约-402 quota-exhausted 与 filtered 区分",
            env_402.failure is not None
            and env_402.failure.reason_code == ReasonCode.QUOTA_EXHAUSTED
            and env_flt.failure is not None
            and env_flt.failure.reason_code == ReasonCode.FILTERED,
        )

        # E12: MCP 8 大正交工具完备性与 structuredContent
        srv = OmniMCPServer(health_path=pathlib.Path(td) / "h4.json")
        tl = await srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        tnames = {t["name"] for t in tl["result"]["tools"]}
        expected_tools = {
            "omni_search",
            "omni_fetch",
            "omni_batch",
            "omni_research",
            "omni_index",
            "omni_doctor",
            "omni_monitor",
            "omni_verify",
        }
        record("E12", "MCP-8 正交工具及 outputSchema 完备", expected_tools == tnames and all("outputSchema" in t for t in tl["result"]["tools"]))

        # E13: 多模式研究编排端到端（离线：feasibility + 证据分级 + Markdown 简报）
        async def _mock_fetch(url: str, max_chars: int = 2000, timeout_sec: float = 5.0):
            from omni_search.fetcher import FetchResult
            return FetchResult(ok=True, url=url, status_code=200, title="Doc", content="Authoritative text", untrusted_content=True)

        rep = await run_research_rounds(
            pool_rrf,
            "shared doc research",
            mode="feasibility",
            domains=["web", "knowledge"],
            limit=3,
            verify_top_k=1,
            render_markdown=True,
            fetcher=_mock_fetch,
        )
        record(
            "E13",
            "编排-omni_research 多模式证据分级与简报渲染",
            len(rep.candidates) == 2
            and len(rep.verified_evidence) == 1
            and rep.verified_evidence[0].verified
            and bool(rep.markdown and rep.plan_id in rep.markdown),
        )

        # E14: 本地多模态 FTS5 索引（中英文 + 增量跳过）
        from omni_search.local_index import LocalIndexEngine, index_paths

        sample_dir = pathlib.Path(td) / "local_ws"
        sample_dir.mkdir()
        (sample_dir / "设计.md").write_text("本地多模态搜索引擎架构说明", encoding="utf-8")
        (sample_dir / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        ldb = pathlib.Path(td) / "idx.db"
        st1 = index_paths([sample_dir], ldb)
        st2 = index_paths([sample_dir], ldb)
        le = LocalIndexEngine(ldb)
        env_l = await le.search("多模态搜索", 5, None)
        record(
            "E14",
            "本地-FTS5 增量索引与中文/多模态检索",
            st1.indexed == 2 and st2.skipped_unchanged == 2 and len(env_l.results) == 1 and env_l.results[0].modality == "document",
        )

        # E15: 多 Agent 工单导出 + 严格合单（拒收错 plan_id）+ 时效冲突审计
        from omni_search.research import audit_freshness_and_conflicts, export_research_workorders, merge_lane_results

        wdir = pathlib.Path(td) / "workorders"
        plan_w = export_research_workorders(wdir, "ai agent search", mode="factcheck")
        ln0 = plan_w.lanes[0]
        good_resp = await pool_rrf.search(ln0.query, domains=["web", "knowledge"])
        (wdir / "agents" / f"{ln0.name}.result.json").write_text(
            json.dumps({"plan_id": plan_w.plan_id, "lane": ln0.name, "query": ln0.query, "domains": ln0.domains, "response": good_resp.model_dump()}),
            encoding="utf-8",
        )
        (wdir / "agents" / "bad.result.json").write_text(
            json.dumps({"plan_id": "wrong000", "lane": ln0.name, "query": ln0.query, "domains": ln0.domains, "response": good_resp.model_dump()}),
            encoding="utf-8",
        )
        merged_rep = await merge_lane_results(wdir, verify_top_k=0)
        fa_check = audit_freshness_and_conflicts(
            [SearchItem(id="o1", domain="academic", source="arxiv", title="Old", url="http://arxiv.org/abs/2001.00001")],
            current_year=2026,
        )
        record(
            "E15",
            "编排-多 Agent 严审合单与时效冲突审计",
            merged_rep.merge_audit is not None
            and len(merged_rep.merge_audit.rejected) == 1
            and ln0.name in merged_rep.merge_audit.accepted_lanes
            and fa_check.stale_count == 1,
        )

        # E16: 本地 SQLite 混合检索（FTS5 + Vector RRF）与 PDF 正文抽取
        hybrid_ws = pathlib.Path(td) / "hybrid_ws"
        hybrid_ws.mkdir()
        (hybrid_ws / "federation.md").write_text(
            "Federated Metasearch Pool implements reciprocal rank fusion and circuit breaker.",
            encoding="utf-8",
        )
        stream = b"BT /F1 12 Tf 72 720 Td (SQLite Hybrid PDF Evidence 2026) Tj ET"
        pdf_objs = [
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n",
        ]
        buf = bytearray(b"%PDF-1.4\n")
        offs = [0]
        for o in pdf_objs:
            offs.append(len(buf))
            buf.extend(o)
        xp = len(buf)
        buf.extend(f"xref\n0 {len(offs)}\n0000000000 65535 f \n".encode("latin-1"))
        for off in offs[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
        buf.extend(f"trailer << /Size {len(offs)} /Root 1 0 R >>\nstartxref\n{xp}\n%%EOF\n".encode("latin-1"))
        (hybrid_ws / "evidence.pdf").write_bytes(bytes(buf))

        hdb = pathlib.Path(td) / "hybrid.db"
        hst = index_paths([hybrid_ws], db_path=hdb)
        heng = LocalIndexEngine(db_path=hdb)
        h_pdf = await heng.search("Hybrid PDF Evidence", limit=5)
        h_vec = await heng.search("federating metasearching reciprocally", limit=5)
        record(
            "E16",
            "本地-SQLite FTS5+向量混合检索与 PDF 正文抽取",
            hst.vectors_updated == 2
            and hst.metadata_only == 0
            and any(r.title == "evidence.pdf" and not r.extra.get("metadata_only") for r in h_pdf.results)
            and any(r.title == "federation.md" and r.extra.get("retrieval_mode") == "vector" for r in h_vec.results),
            f"vecs={hst.vectors_updated} pdf_hits={len(h_pdf.results)} vec_hits={len(h_vec.results)}",
        )

        # E17: 安全加固-逐跳 SSRF 重定向拦截 + 公网 PDF 直抽 + BibTeX 引用导出
        from omni_search.research import ResearchReport, render_citations_bibtex

        redir_calls = 0

        def redir_handler(req: httpx.Request) -> httpx.Response:
            nonlocal redir_calls
            redir_calls += 1
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/internal"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(redir_handler)) as cli_r:
            fr_redir = await safe_fetch_url("https://example.org/jump", client=cli_r, resolve_host=False)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=bytes(buf))
            )
        ) as cli_pdf:
            fr_pdf = await safe_fetch_url("https://arxiv.org/pdf/2608.24060v1", client=cli_pdf, resolve_host=False)

        bib_rep = ResearchReport(
            version="0.1.0",
            plan_id="eval17bib",
            brief="eval bib",
            mode="standard",
            lanes=[],
            candidates=[
                SearchItem(
                    id="b1",
                    domain="academic",
                    source="arxiv",
                    title="SQLite Hybrid Search",
                    url="https://arxiv.org/abs/2608.24060v1",
                    published_at="2026-08-20",
                    extra={"evidence_tier": "T1", "pub_year": 2026},
                )
            ],
        )
        bib_txt = render_citations_bibtex(bib_rep)
        record(
            "E17",
            "安全与文献-逐跳 SSRF 拦截 + PDF URL 抽取 + BibTeX 导出",
            (fr_redir.ok is False and fr_redir.reason_code == "ssrf-blocked" and redir_calls == 1)
            and (fr_pdf.ok is True and "SQLite Hybrid PDF Evidence 2026" in fr_pdf.content)
            and ("@article{" in bib_txt and "eprint = {2608.24060}" in bib_txt),
            f"redir_calls={redir_calls} pdf_ok={fr_pdf.ok}",
        )

        # E18: 内外联合调研 (include_local) + 确定性自反追问 (deepen) + file:// 隐私隔离
        from omni_search.research import extract_followup_terms, plan_research_lanes

        p_loc = plan_research_lanes("sqlite hybrid search", domains=["academic"], include_local=True)
        ft = extract_followup_terms(
            [
                SearchItem(id="f1", domain="academic", source="arxiv", title="scrydb sqlite-vec bm25", url="https://arxiv.org/abs/2608.24060", snippet="sqlite-vec with bm25"),
                SearchItem(id="f2", domain="code", source="github_direct", title="sqlite-vec", url="https://github.com/asg017/sqlite-vec", snippet="bm25 hybrid"),
            ],
            brief="sqlite hybrid search",
            max_terms=2,
        )
        bib_local_rep = ResearchReport(
            version="0.1.0",
            plan_id="eval18loc",
            brief="sqlite",
            mode="standard",
            lanes=[],
            candidates=[
                SearchItem(id="l1", domain="local", source="local_index", title="secret.md", url="file:///D:/secret.md"),
                SearchItem(id="w1", domain="academic", source="arxiv", title="Public Paper", url="https://arxiv.org/abs/2608.24060"),
            ],
        )
        bib_clean = render_citations_bibtex(bib_local_rep)
        record(
            "E18",
            "编排-内外联合调研(local_kb) + 确定性自反提词 + file:// 隐私隔离",
            any(ln.name == "local_kb" for ln in p_loc.lanes)
            and ("sqlite-vec" in ft and "bm25" in ft)
            and ("file://" not in bib_clean and "2608.24060" in bib_clean),
            f"terms={ft}",
        )

        # E19: omni_fetch v2 结构化 Markdown + npm/GitHub 官方 API 映射 + 畸形 URL 入池拦截
        md_html = "<html><head><title>Doc</title></head><body><main><h1>Title</h1><ul><li>Item A</li></ul><pre><code>x = 1</code></pre><table><tr><th>K</th><th>V</th></tr><tr><td>a</td><td>1</td></tr></table></main></body></html>"
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=md_html))
        ) as cli_md:
            f_md = await safe_fetch_url("https://docs.example.com/md", client=cli_md, resolve_host=False)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={"name": "demo-pkg", "dist-tags": {"latest": "2.0.0"}, "readme": "# demo-pkg\n\nClean npm readme."},
                )
                if req.url.host == "registry.npmjs.org"
                else httpx.Response(403, text="Forbidden")
            )
        ) as cli_npm:
            f_npm = await safe_fetch_url("https://www.npmjs.com/package/demo-pkg", client=cli_npm, resolve_host=False)

        record(
            "E19",
            "读取-结构化 Markdown 抽取 + npm 官方 Registry API 映射",
            (f_md.ok and "# Title" in f_md.content and "- Item A" in f_md.content and "| K | V |" in f_md.content)
            and (f_npm.ok and f_npm.source_via == "npm-registry-api" and "Clean npm readme." in f_npm.content),
            f"md_ok={f_md.ok} npm_via={f_npm.source_via}",
        )

        # E20: 本地长文档分块向量召回 + Model2Vec 工厂适配
        from omni_search.embeddings import Model2VecEmbedder, build_default_embedder

        chunk_ws = pathlib.Path(td) / "chunk_ws"
        chunk_ws.mkdir()
        long_lines = [f"Filler ledger entry number {i} for warehouse box alpha." for i in range(1, 130)]
        long_lines.append("## Resilience Section")
        long_lines.append("Adaptive circuit breaker exponential backoff prevents cascading failures.")
        (chunk_ws / "long_manual.md").write_text("\n".join(long_lines), encoding="utf-8")
        cdb = pathlib.Path(td) / "chunk.db"
        cst = index_paths([chunk_ws], db_path=cdb)
        ceng = LocalIndexEngine(db_path=cdb)
        cres = await ceng.search("breakers exponentially backoffs", limit=5)
        m2v_built = build_default_embedder({"OMNI_EMBED_BACKEND": "model2vec", "OMNI_EMBED_MODEL": "demo/potion"})
        record(
            "E20",
            "本地-分块(Chunk)向量深处召回 + Model2Vec 适配器工厂",
            cst.chunks_indexed >= 4
            and any(r.title == "long_manual.md" and r.extra.get("retrieval_mode") == "vector" and str(r.extra.get("matched_locator", "")).startswith("L") for r in cres.results)
            and isinstance(m2v_built, Model2VecEmbedder),
            f"chunks={cst.chunks_indexed} hits={len(cres.results)}",
        )

        # E21: 跨源精排降级韧性 + 调研历史快照对比 (diff_research_reports)
        from omni_search.reranker import build_default_reranker
        from omni_search.research import diff_research_reports

        rr_cfg = build_default_reranker({"OMNI_RERANK_URL": "http://127.0.0.1:8000/v1", "OMNI_RERANK_MODEL": "bge-reranker"})
        rep_a = ResearchReport(
            version="0.1.0",
            plan_id="a1",
            brief="demo",
            mode="standard",
            lanes=[],
            tier_counts={"T1": 1, "T2": 0, "T3": 0},
            candidates=[SearchItem(id="1", domain="academic", source="arxiv", title="Paper v1", url="http://arxiv.org/abs/2608.24060v1", extra={"evidence_tier": "T1"})],
        )
        rep_b = ResearchReport(
            version="0.1.0",
            plan_id="b2",
            brief="demo",
            mode="standard",
            lanes=[],
            tier_counts={"T1": 2, "T2": 0, "T3": 0},
            candidates=[
                SearchItem(id="1", domain="academic", source="arxiv", title="Paper v2", url="https://arxiv.org/pdf/2608.24060v2", extra={"evidence_tier": "T1"}),
                SearchItem(id="2", domain="code", source="github_direct", title="New Repo", url="https://github.com/o/new", extra={"evidence_tier": "T1"}),
            ],
        )
        df_ab = diff_research_reports(rep_a, rep_b)
        record(
            "E21",
            "精排与追踪-Reranker 工厂 + 历史快照规范 URL 变更对比",
            (rr_cfg is not None and rr_cfg.id == "http_rerank:bge-reranker")
            and (df_ab.retained_count == 1 and len(df_ab.added) == 1 and df_ab.tier_delta["T1"] == 1),
            f"retained={df_ab.retained_count} added={len(df_ab.added)}",
        )

        # E22: 100% 原生替代代码引擎（无 mcps 桥接 + grep.app SSE 解析 + 免建库 rg 实时检索）
        from omni_search.code_evidence import parse_grep_app_text
        from omni_search.local_code import LocalCodeEngine
        from omni_search.mcp_server import build_default_engines

        nat_ids = {e.id for e in build_default_engines(env={})}
        ga_items = parse_grep_app_text(
            "Repository: org/demo\nPath: a.py\nURL: https://github.com/org/demo/blob/main/a.py\n--- Snippet 1 (Line 7) ---\ndef hi(): pass\n",
            limit=3,
        )
        rg_ws = pathlib.Path(td) / "rg_ws"
        rg_ws.mkdir()
        (rg_ws / "demo.py").write_text("def eval22_unique_fn():\n    return 1\n", encoding="utf-8")
        rg_env = await LocalCodeEngine(roots=[rg_ws]).search("lang:py eval22_unique_fn", limit=3)
        record(
            "E22",
            "原生代码替代-50 原生引擎(零桥接) + grep.app 解析 + rg 实时检索",
            ("mcps_repo" not in nat_ids and len(nat_ids) == 50)
            and (len(ga_items) == 1 and ga_items[0].extra.get("line") == 7)
            and (rg_env.failure is None and len(rg_env.results) == 1 and rg_env.results[0].extra.get("line") == 1),
            f"native_count={len(nat_ids)} rg_hits={len(rg_env.results)}",
        )

        # E23: 第 11 域 security 路由 + 41 声明式官方信源完备性
        record(
            "E23",
            "信源扩展-11 域(含 security)路由与 41 声明式官方 YAML 校验",
            "security" in route_domains("CVE-2026-9999 sqlite 漏洞", ["auto"])
            and {"nvd_cve", "ietf_rfc", "zenodo", "europe_pmc", "hf_datasets", "wikimedia_commons", "github_issues", "anaconda", "terraform_registry", "google_news_rss"}.issubset(nat_ids),
            f"total_native={len(nat_ids)}",
        )

        # E24: 性能与质量精修-WAL+BM25F、跨进程磁盘缓存、连接池复用与省 Token 精简视图
        import sqlite3
        from omni_search.cache import DiskCacheStore
        from omni_search.contracts import compact_search_response
        from omni_search.research import classify_evidence_tier

        wal_conn = sqlite3.connect(str(cdb))
        try:
            jmode = str(wal_conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        finally:
            wal_conn.close()
        dc_path = pathlib.Path(td) / "eval_cache.db"
        dc = DiskCacheStore(dc_path, ttl_sec=120.0)
        dc.set("k1", '{"ok":true}')
        concise_sample = compact_search_response(good_resp)
        record(
            "E24",
            "性能与质量-WAL并发 + 跨进程磁盘缓存 + concise省Token + dockerhub分级",
            jmode == "wal"
            and dc.get("k1") == '{"ok":true}'
            and concise_sample.get("concise") is True
            and classify_evidence_tier("https://hub.docker.com/r/a/b", source="dockerhub") == "T2",
            f"journal={jmode}",
        )

        # E25: 纯 Python 多模态媒体头元数据解析 + 分辨率/时长检索 + Google News 跳转链降权
        import struct
        from omni_search.media_meta import extract_media_metadata
        from omni_search.research import _select_diverse_verify_targets

        mws = pathlib.Path(td) / "media_eval"
        mws.mkdir()
        ihdr = struct.pack(">IIBBBBB", 1920, 1080, 8, 2, 0, 0, 0)
        png_bytes = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
        mvhd_body = struct.pack(">IIIIII", 0, 0, 0, 1000, 42000, 0) + b"\x00" * 76
        mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
        moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
        mp4_bytes = struct.pack(">I", 16) + b"ftyp" + b"isom\x00\x00\x00\x00" + moov
        (mws / "slide.png").write_bytes(png_bytes)
        (mws / "clip.mp4").write_bytes(mp4_bytes)
        mdb = pathlib.Path(td) / "media_eval.db"
        index_paths([mws], db_path=mdb)
        meng = LocalIndexEngine(db_path=mdb)
        mr_img = await meng.search("type:image 1920x1080", limit=3)
        mr_vid = await meng.search("type:video 42.0s", limit=3)
        gn_cand = SearchItem(
            id="gn",
            domain="news",
            source="google_news_rss",
            title="GN",
            url="https://news.google.com/rss/articles/CBMiTEST",
            score=0.9,
            extra={"evidence_tier": "T3"},
        )
        dir_cand = SearchItem(
            id="dir",
            domain="news",
            source="hackernews",
            title="Direct",
            url="https://example.org/post",
            score=0.3,
            extra={"evidence_tier": "T3"},
        )
        picked_v = _select_diverse_verify_targets([gn_cand, dir_cand], top_k=1)
        record(
            "E25",
            "多模态与核验-纯Python媒体头解析(PNG/MP4) + 分辨率/时长检索 + GoogleNews跳转降权",
            extract_media_metadata(mws / "slide.png").get("resolution") == "1080p"
            and len(mr_img.results) == 1
            and mr_img.results[0].title == "slide.png"
            and len(mr_vid.results) == 1
            and mr_vid.results[0].title == "clip.mp4"
            and len(picked_v) == 1
            and picked_v[0].url == dir_cand.url,
            f"img={len(mr_img.results)} vid={len(mr_vid.results)} top_verify={picked_v[0].url if picked_v else ''}",
        )

        # E26: Phase 2A 质量跃升 - 智能提词 + Fast-Fail 分级熔断 + <svg><title> 防污染
        from omni_search.fetcher import _CleanTextExtractor
        from omni_search.research import widen_query

        wq_e26 = widen_query("AI agent web search MCP server markdown extraction reranking", max_terms=4).lower()
        hs_e26 = HealthStore(pathlib.Path(td) / "h_e26.json")
        hs_e26.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=2000.0)
        st_e26 = hs_e26.check("deepseek", now=2010.0)
        ex_e26 = _CleanTextExtractor()
        ex_e26.feed("<html><head><title>Doc | MDN</title></head><body><header><svg><title>MDN</title></svg></header><article><p>" + ("clean body " * 20) + "</p></article></body></html>")
        record(
            "E26",
            "质量与韧性-高信息量提词 + 持久故障Fast-Fail熔断 + SVG标题隔离与Article优先",
            ("mcp" in wq_e26 and "markdown" in wq_e26 and "reranking" in wq_e26)
            and (st_e26.in_cooldown and st_e26.skip_fast)
            and ("".join(ex_e26.title_parts).strip() == "Doc | MDN" and len("".join(ex_e26.article_parts)) >= 120),
            f"widen='{wq_e26}' skip_fast={st_e26.skip_fast}",
        )

        # E27: Phase 2B 本地知识库专业化 - SHA-256 去重 + collection/tag 隔离 + 一跳链接图谱
        import os as _os

        kb_dir = pathlib.Path(td) / "kb_e27"
        kb_dir.mkdir()
        p_a = kb_dir / "hub.md"
        p_b = kb_dir / "target_spec.md"
        p_a.write_text("---\ntags: [rag, core]\n---\n\nHubE27Anchor references [[target_spec]].\n", encoding="utf-8")
        p_b.write_text("# Target Spec\n\nTarget body.\n", encoding="utf-8")
        kb_db = pathlib.Path(td) / "kb_e27.db"
        s_init = index_paths([kb_dir], db_path=kb_db, collection="work")
        st_a = p_a.stat()
        _os.utime(p_a, ns=(st_a.st_atime_ns, st_a.st_mtime_ns + 2_000_000_000))
        s_touch = index_paths([kb_dir], db_path=kb_db, collection="work")
        r_e27 = await LocalIndexEngine(db_path=kb_db).search("collection:work tag:rag HubE27Anchor", limit=3)
        rel_e27 = {d["title"] for d in (r_e27.results[0].extra.get("related_docs") or [])} if r_e27.results else set()
        record(
            "E27",
            "本地KB专业化-SHA256哈希去重 + collection/tag隔离 + 一跳链接图谱",
            s_init.indexed == 2
            and s_touch.indexed == 0
            and s_touch.vectors_updated == 0
            and len(r_e27.results) == 1
            and "target_spec.md" in rel_e27,
            f"touch_idx={s_touch.indexed} rel={sorted(rel_e27)}",
        )

        # E28: Phase 2C 效费比遥测 + IR 黄金基准质量门禁
        root_dir = str(pathlib.Path(__file__).resolve().parents[1])
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from omni_search.telemetry import TelemetryStore
        from evals.ir_benchmark import run_offline_ir_benchmark

        tstore = TelemetryStore(pathlib.Path(td) / "tel_e28.db")
        tstore.record_batch(
            [
                {"engine": "good_a", "domain": "web", "ok": True, "reason_code": "ok", "elapsed_ms": 150, "result_count": 5, "top5_hits": 2, "skipped_fast": False},
                {"engine": "slow_b", "domain": "news", "ok": False, "reason_code": "deadline", "elapsed_ms": 7200, "result_count": 0, "top5_hits": 0, "skipped_fast": False},
                {"engine": "slow_b", "domain": "news", "ok": False, "reason_code": "deadline", "elapsed_ms": 7400, "result_count": 0, "top5_hits": 0, "skipped_fast": False},
                {"engine": "slow_b", "domain": "news", "ok": False, "reason_code": "deadline", "elapsed_ms": 7600, "result_count": 0, "top5_hits": 0, "skipped_fast": False},
            ]
        )
        tsum = tstore.summarize(window_days=7)
        ir_res = await run_offline_ir_benchmark()
        record(
            "E28",
            "可观测与IR基准-SQLite效费比遥测(P95/Top5/建议) + 14题IR黄金基准(MRR/Noise)",
            tsum["total_events"] == 4
            and any("slow_b" in rec for rec in tsum["recommendations"])
            and ir_res["pass"] is True
            and ir_res["metrics"]["mrr"] >= 0.85,
            f"mrr={ir_res['metrics']['mrr']} hit3={ir_res['metrics']['hit_at_3']} noise3={ir_res['metrics']['noise_at_3']} save={ir_res['metrics']['concise_saving_ratio']}",
        )

        # E29: Phase 2D MCP Resources/Prompts/Progress + omni monitor 增量快照追踪
        from omni_search.monitor import run_monitor_check

        notifs_e29: list[dict[str, Any]] = []
        srv29 = OmniMCPServer(health_path=pathlib.Path(td) / "h29.json", emit_notification=notifs_e29.append)
        init29 = await srv29.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        rlist29 = await srv29.handle_request({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        plist29 = await srv29.handle_request({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})
        await srv29.aclose()

        mon_items = [SearchItem(id="m1", domain="academic", source="arxiv", title="Base", url="https://arxiv.org/abs/2608.00001", snippet="base")]

        class _MonEng:
            id = "arxiv"
            domain = "academic"
            tier = "L0"
            deadline_sec = 3.0

            async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
                return EngineEnvelope(engine=self.id, domain=self.domain, results=list(mon_items))

        mpool = FederatedPool(engines=[_MonEng()], health_store=HealthStore(pathlib.Path(td) / "hm.json"), cache_ttl_sec=0.0)
        mdir = pathlib.Path(td) / "mon"
        mr1 = await run_monitor_check(mpool, "E29 Topic", state_dir=mdir, domains=["academic"], verify_top_k=0)
        mon_items.append(SearchItem(id="m2", domain="academic", source="arxiv", title="New T1", url="https://arxiv.org/abs/2611.00002", snippet="new"))
        mr2 = await run_monitor_check(mpool, "E29 Topic", state_dir=mdir, domains=["academic"], verify_top_k=0)
        await mpool.aclose()
        record(
            "E29",
            "MCP协议与追踪-Resources/Prompts/Progress + monitor增量快照变更检出",
            "resources" in init29["result"]["capabilities"]
            and "prompts" in init29["result"]["capabilities"]
            and len(rlist29["result"]["resources"]) >= 3
            and len(plist29["result"]["prompts"]) >= 3
            and mr1.is_baseline is True
            and mr2.has_changes is True
            and len(mr2.added_t1) == 1,
            f"res={len(rlist29['result']['resources'])} prompts={len(plist29['result']['prompts'])} added_t1={len(mr2.added_t1)}",
        )

        # E30: Phase 3A 50 原生引擎（PyPI / Go / HF Papers / OSV.dev 双源安全）+ 泛词精修
        wq30 = widen_query("deep research agentic search contextual retrieval late chunking 2026", max_terms=4).lower()
        record(
            "E30",
            "v0.3信源扩展-50原生引擎(PyPI/Go/HFPapers/OSV双安全源) + 泛词防霸榜",
            len(nat_ids) == 50
            and {"pypi_ecosystems", "golang_ecosystems", "hf_papers", "osv_vulns"}.issubset(nat_ids)
            and "research" not in wq30
            and "contextual" in wq30
            and "chunking" in wq30,
            f"native={len(nat_ids)} widen='{wq30}'",
        )

        # E31: Phase 3B 长文查询聚焦裁剪 + 受控站点目录映射（SSRF 防护 + llms.txt/sitemap 融合）
        from omni_search.fetcher import prune_content_for_query
        from omni_search.site_map import safe_map_site

        long_doc = "\n\n".join(
            ["# Handbook"] + [f"Background filler paragraph {i} with generic prose." for i in range(40)] + ["UniqueAnchorMetrics saves 72% tokens."]
        )
        pruned_txt, did_prune = prune_content_for_query(long_doc, "UniqueAnchorMetrics tokens", max_chars=240)
        ssrf_map = await safe_map_site("http://127.0.0.1/secret", resolve_host=False)
        record(
            "E31",
            "站点映射与聚焦裁剪-长文Query聚焦段裁剪 + safe_map_site SSRF阻断",
            did_prune is True
            and "UniqueAnchorMetrics saves 72% tokens." in pruned_txt
            and len(pruned_txt) <= 240
            and ssrf_map.ok is False
            and ssrf_map.reason_code == "ssrf-blocked",
            f"pruned_len={len(pruned_txt)} ssrf={ssrf_map.reason_code}",
        )

        # E32: Phase 3C 零 LLM Contextual Chunking 层级面包屑 + news/security 时效新鲜度加权
        from omni_search.local_index import _split_line_chunks
        from omni_search.pool import fuse_results_rrf

        md_ctx_txt = "# CoreArch\n\n## Breaker\n\n" + "\n".join(f"filler line {i}" for i in range(42)) + "\ndeep threshold rule\n"
        ctx_chunks = _split_line_chunks(md_ctx_txt)
        env_old = EngineEnvelope(
            engine="s1",
            domain="news",
            results=[SearchItem(id="o", domain="news", source="s1", title="Topic Update", url="https://a.example/old", snippet="topic update", published_at="2015-01-01")],
        )
        env_new = EngineEnvelope(
            engine="s2",
            domain="news",
            results=[SearchItem(id="n", domain="news", source="s2", title="Topic Update", url="https://b.example/new", snippet="topic update", published_at="2026-10-01")],
        )
        fused_rec = fuse_results_rrf([env_old, env_new], query="Topic Update")
        record(
            "E32",
            "语义与时效-零LLM层级面包屑ContextualChunking + news/security鲜度加权",
            len(ctx_chunks) >= 2
            and "CoreArch > Breaker" in ctx_chunks[-1][1]
            and fused_rec[0].id == "n",
            f"chunks={len(ctx_chunks)} top_news={fused_rec[0].id}",
        )

        # E33: Phase 3D MCP Streamable HTTP 本机回环服务（Bearer 401 + Origin 403 + /health + /mcp）
        from omni_search.mcp_http import McpHttpServer

        mcp33 = OmniMCPServer(health_path=pathlib.Path(td) / "h33.json")
        non_loop_blocked = False
        try:
            McpHttpServer(mcp33, host="0.0.0.0", port=0, auth_token="")
        except ValueError:
            non_loop_blocked = True
        hsrv = McpHttpServer(mcp33, host="127.0.0.1", port=0, auth_token="e33-tok")
        await hsrv.start()
        try:
            async with httpx.AsyncClient() as hcli:
                h_ok = await hcli.get(f"http://127.0.0.1:{hsrv.port}/health")
                h_401 = await hcli.post(f"http://127.0.0.1:{hsrv.port}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
                h_200 = await hcli.post(
                    f"http://127.0.0.1:{hsrv.port}/mcp",
                    headers={"Authorization": "Bearer e33-tok"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                )
        finally:
            await hsrv.stop()
            await mcp33.aclose()
        record(
            "E33",
            "常驻共享传输-MCP Streamable HTTP回环保护 + Bearer鉴权 + JSON-RPC握手",
            non_loop_blocked
            and h_ok.status_code == 200
            and h_401.status_code == 401
            and h_200.status_code == 200
            and h_200.json()["result"]["serverInfo"]["name"] == "search-all-in-one",
            f"port={hsrv.port} health={h_ok.status_code} unauth={h_401.status_code}",
        )

        # E34: Phase 4A 半开熔断冷却到期收紧探测超时 + strip_html HTML 实体解码清洗
        from omni_search.declarative import _apply_transform

        hs34 = HealthStore(pathlib.Path(td) / "h34.json", probe_deadline_sec=3.5)
        for idx in range(5):
            hs34.record_failure("gdelt", ReasonCode.DEADLINE, now=1000.0 + idx)
        cd34 = hs34.check("gdelt", now=1010.0)
        ho34 = hs34.check("gdelt", now=cd34.cooldown_until + 1.0)
        cleaned34 = _apply_transform("<b>Top&nbsp;&nbsp;APIs</b> &amp; &#39;2026&#39;", "strip_html")
        record(
            "E34",
            "可靠性与纯净度-冷却到期半开探测收紧 + HTML实体解码清洗",
            cd34.skip_fast is True
            and ho34.half_open is True
            and ho34.probe_deadline_sec == 3.5
            and cleaned34 == "Top APIs & '2026'",
            f"half_open={ho34.half_open} probe={ho34.probe_deadline_sec}s cleaned={cleaned34!r}",
        )

        # E35: Phase 4B 批量安全抓取 SSRF 隔离 + MCP omni_monitor / omni_verify 工具暴露
        from omni_search.fetcher import safe_fetch_urls_batch

        bf35 = await safe_fetch_urls_batch(
            urls=["http://127.0.0.1:8080/a", "http://127.0.0.1:8080/a", "http://169.254.169.254/latest/meta-data/"],
            resolve_host=False,
        )
        mcp35 = OmniMCPServer(health_path=pathlib.Path(td) / "h35.json")
        try:
            tlist35 = await mcp35.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            tnames35 = {t["name"] for t in tlist35["result"]["tools"]}
        finally:
            await mcp35.aclose()
        record(
            "E35",
            "批量抓取与MCP补齐-safe_fetch_urls_batch去重隔离 + omni_monitor/omni_verify暴露",
            bf35.total == 2
            and bf35.failed == 2
            and bf35.untrusted_content is True
            and {"omni_monitor", "omni_verify"}.issubset(tnames35),
            f"batch_total={bf35.total} mcp_tools={len(tnames35)}",
        )

        # E36: Phase 4C 主张级引用核验引擎（supported / partial / unsupported / blocked 精确判定）
        from omni_search.fetcher import FetchResult
        from omni_search.verifier import ClaimInput, verify_claims

        async def _stub_vfetch(url: str, max_chars: int = 12000, timeout_sec: float = 15.0) -> FetchResult:
            if "fts5" in url:
                return FetchResult(ok=True, url=url, status_code=200, title="SQLite FTS5", content="SQLite FTS5 supports BM25 ranking and prefix queries.", untrusted_content=True)
            if "cooking" in url:
                return FetchResult(ok=True, url=url, status_code=200, title="Pasta", content="Boil pasta with salt and olive oil.", untrusted_content=True)
            return FetchResult(ok=False, url=url, reason_code="ssrf-blocked", reason="private ip", untrusted_content=True)

        vrep36 = await verify_claims(
            claims=[
                ClaimInput(claim="SQLite FTS5 supports BM25 ranking", url="https://docs.example.com/fts5"),
                ClaimInput(claim="CVE-2026-9999 remote code execution in OpenSSL", url="https://docs.example.com/cooking"),
                ClaimInput(claim="Internal service leak", url="http://127.0.0.1/secret"),
            ],
            render_markdown=True,
            fetcher=_stub_vfetch,
        )
        record(
            "E36",
            "防幻觉引用核验-主张级证据对齐 (supported/unsupported/blocked)",
            vrep36.supported_count == 1
            and vrep36.unsupported_count == 1
            and vrep36.blocked_count == 1
            and vrep36.untrusted_content is True
            and bool(vrep36.markdown and "引用核验审计报告" in vrep36.markdown),
            f"supported={vrep36.supported_count} unsupported={vrep36.unsupported_count} blocked={vrep36.blocked_count}",
        )

        # E37: Phase 5A 长查询自适应宽词回退（严格 AND 源首轮 0 命中自动宽词补查）
        from omni_search.contracts import RequestStats
        from omni_search.pool import relax_long_query

        class _StrictAndEvalEng:
            id = "strict_e37"
            domain = "code"
            tier = "L0"
            deadline_sec = 5.0
            calls = 0

            def is_ready(self) -> tuple[bool, str]:
                return True, "ok"

            async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
                self.calls += 1
                if len(query.split()) > 4:
                    return EngineEnvelope(engine=self.id, domain=self.domain, results=[], requests=RequestStats(attempted=1, succeeded=1, failed=0))
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    results=[SearchItem(id="r37", domain="code", source=self.id, title="MCP Server", url="https://github.com/modelcontextprotocol/servers", snippet="MCP server")],
                    requests=RequestStats(attempted=1, succeeded=1, failed=0),
                )

        se37 = _StrictAndEvalEng()
        p37 = FederatedPool(engines=[se37], health_store=HealthStore(pathlib.Path(td) / "h37.json"), cache_ttl_sec=0.0)
        try:
            r37 = await p37.search("how to build MCP server search fetch extract crawl 2026", domains=["code"], limit=4, budget_ms=8000, use_cache=False)
        finally:
            await p37.aclose()
        rq37 = relax_long_query("how to build MCP server search fetch extract crawl 2026", max_terms=4)
        record(
            "E37",
            "长查询鲁棒性-FederatedPool自适应宽词补查与告警披露",
            se37.calls == 2 and len(r37.results) == 1 and len(rq37.split()) <= 4 and any("宽词补查" in w for w in r37.warnings),
            f"calls={se37.calls} relaxed={rq37!r}",
        )

        # E38: Phase 5B omni_research 自动引用对齐审计（识别跑题低对齐页面）
        from omni_search.research import VerifiedEvidence, audit_verified_citations

        ca38 = audit_verified_citations(
            "SQLite FTS5 hybrid vector search",
            [
                VerifiedEvidence(url="https://arxiv.org/abs/2608.00001", title="FTS5", verified=True, excerpt="SQLite FTS5 hybrid vector search combines BM25F and RRF."),
                VerifiedEvidence(url="https://cook.example.org/pasta", title="Pasta", verified=True, excerpt="Boil tomato basil pasta with olive oil."),
            ],
        )
        record(
            "E38",
            "调研闭环-ResearchReport自动主张级引用对齐审计",
            ca38 is not None and ca38.supported_count == 1 and ca38.unsupported_count == 1 and "https://cook.example.org/pasta" in ca38.flagged_urls,
            f"supported={ca38.supported_count if ca38 else 0} unsupported={ca38.unsupported_count if ca38 else 0}",
        )

        # E39: Phase 5C 站内定向采集 safe_crawl_site（map 排序 + 并发抓 Top-N + SSRF 阻断）
        from omni_search.site_map import safe_crawl_site

        def _crawl_handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/llms.txt":
                return httpx.Response(200, headers={"content-type": "text/plain"}, text="- [FTS5 Guide](https://docs.example.com/fts5)\n- [Other](https://docs.example.com/other)\n")
            if req.url.path == "/fts5":
                return httpx.Response(200, headers={"content-type": "text/html"}, text="<html><body><main><p>FTS5 hybrid guide</p></main></body></html>")
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<html><body><main><p>Other page</p></main></body></html>")

        async with httpx.AsyncClient(transport=httpx.MockTransport(_crawl_handler), follow_redirects=False) as ccli39:
            cr39 = await safe_crawl_site("https://docs.example.com", query="FTS5", max_pages=1, client=ccli39, resolve_host=False)
        cr39_ssrf = await safe_crawl_site("http://127.0.0.1:8080/secret", resolve_host=False)
        record(
            "E39",
            "站内定向采集-safe_crawl_site(map+fetch联动)与SSRF拦截",
            cr39.ok is True and cr39.crawled_total == 1 and cr39.pages[0].url == "https://docs.example.com/fts5" and cr39_ssrf.reason_code == "ssrf-blocked",
            f"crawled={cr39.crawled_total} top={cr39.pages[0].url if cr39.pages else ''} ssrf={cr39_ssrf.reason_code}",
        )

        # E40: Phase 6A 融合层信噪比权重 + 零词面重合噪声降权
        e40_iss = EngineEnvelope(
            engine="github_issues",
            domain="code",
            results=[SearchItem(id="n40", domain="code", source="github_issues", title="Unrelated Studio Chat", url="https://github.com/u/r/issues/1", snippet="Digital art studio.")],
        )
        e40_repo = EngineEnvelope(
            engine="github_direct",
            domain="code",
            results=[SearchItem(id="g40", domain="code", source="github_direct", title="firecrawl-mcp-server", url="https://github.com/org/firecrawl-mcp", snippet="Firecrawl MCP server.")],
        )
        r40 = fuse_results_rrf([e40_iss, e40_repo], query="firecrawl MCP server")
        record(
            "E40",
            "排序纯净度-跨源信噪比权重与零重合噪声沉底(不删结果)",
            len(r40) == 2 and r40[0].id == "g40" and r40[-1].extra.get("low_lexical_overlap") is True,
            f"top={r40[0].id} tail_score={r40[-1].score}",
        )

        # E41: Phase 6B research_render 独立模块解耦与 re-export 兼容
        from omni_search import research as _res_mod
        from omni_search import research_render as _rr_mod

        record(
            "E41",
            "架构解耦-research_render独立模块与research零破坏re-export兼容",
            _res_mod.render_research_markdown is _rr_mod.render_research_markdown
            and _res_mod.render_citations_bibtex is _rr_mod.render_citations_bibtex
            and _res_mod.diff_research_reports is _rr_mod.diff_research_reports,
            "re-export=identical",
        )

        # E42: Phase 6C fetcher 代码语言围栏 + 外链保真 + SPA meta 兜底
        def _e42_handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/doc":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text='<html><head><meta property="og:title" content="Spec Doc"/></head><body><article><p>See <a href="https://example.org/ref">ref</a></p><pre><code class="language-python">x = 1</code></pre></article></body></html>',
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<html><head><title>SPA</title><meta name="description" content="SPA meta fallback summary."/></head><body><div id="app"></div></body></html>',
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e42_handler), follow_redirects=False) as c42:
            f42_a = await safe_fetch_url("https://docs.example.com/doc", client=c42, resolve_host=False)
            f42_b = await safe_fetch_url("https://docs.example.com/spa", client=c42, resolve_host=False)
        record(
            "E42",
            "正文保真-代码块语言围栏 + 外链Markdown保真 + SPA meta兜底",
            f42_a.ok
            and f42_a.title == "Spec Doc"
            and "```python" in f42_a.content
            and "[ref](https://example.org/ref)" in f42_a.content
            and f42_b.ok
            and "SPA meta fallback summary" in f42_b.content,
            f"title={f42_a.title!r} spa_ok={f42_b.ok}",
        )

        # E43: Phase 7A local_chunking 独立模块解耦与 local_index re-export 兼容
        from omni_search import local_chunking as _lc_mod
        from omni_search import local_index as _li_mod

        record(
            "E43",
            "架构解耦-local_chunking独立模块与local_index零破坏re-export兼容",
            _li_mod._split_line_chunks is _lc_mod._split_line_chunks
            and _li_mod.parse_local_query is _lc_mod.parse_local_query,
            "re-export=identical",
        )

        # E44: Phase 7B omni doctor 健康汇总 + 半开状态可见 + 可执行修复建议
        mcp44 = OmniMCPServer(health_path=pathlib.Path(td) / "h44.json", env={})
        try:
            mcp44.health_store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=10**12)
            for idx in range(5):
                mcp44.health_store.record_failure("gdelt", ReasonCode.DEADLINE, now=1000.0 + idx)
            doc44 = await mcp44.doctor(probe=False)
        finally:
            await mcp44.aclose()
        record(
            "E44",
            "自愈诊断-omni doctor健康汇总 + half_open可见 + 可执行修复建议",
            doc44["health_summary"]["cooldown_count"] >= 1
            and doc44["health_summary"]["half_open_count"] >= 1
            and any("deepseek" in r for r in doc44["recommendations"]),
            f"cooldown={doc44['health_summary']['cooldown_count']} half_open={doc44['health_summary']['half_open_count']} recs={len(doc44['recommendations'])}",
        )

        # E45: Phase 7C MCP omni_search autoRelax=False 精确检索透传
        mcp45 = OmniMCPServer(health_path=pathlib.Path(td) / "h45.json", env={})
        se45 = _StrictAndEvalEng()
        mcp45.pool = FederatedPool(engines=[se45], cache_ttl_sec=0.0)
        try:
            await mcp45.handle_request(
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
        finally:
            await mcp45.aclose()
        record(
            "E45",
            "精确检索控制-MCP omni_search autoRelax=False禁用宽词补查",
            se45.calls == 1,
            f"calls={se45.calls}",
        )

        # E46: Phase 8A html_markdown 独立模块解耦与 fetcher re-export 兼容
        from omni_search import fetcher as _ft_mod
        from omni_search import html_markdown as _hm_mod

        record(
            "E46",
            "架构解耦-html_markdown独立模块与fetcher零破坏re-export兼容",
            _ft_mod._CleanTextExtractor is _hm_mod._CleanTextExtractor
            and _ft_mod.prune_content_for_query is _hm_mod.prune_content_for_query,
            "re-export=identical",
        )

        # E47: Phase 8B arXiv 与 crates.io 官方 API 直取
        def _e47_handler(request: httpx.Request) -> httpx.Response:
            u = str(request.url)
            if "export.arxiv.org/api/query" in u:
                return httpx.Response(
                    200,
                    text=(
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<feed xmlns="http://www.w3.org/2005/Atom">'
                        "<entry><title>Eval Paper</title><published>2026-01-02T00:00:00Z</published>"
                        "<summary>Auditable retrieval eval abstract.</summary>"
                        "<author><name>Jane Doe</name></author></entry></feed>"
                    ),
                    headers={"content-type": "application/atom+xml"},
                )
            if u == "https://crates.io/api/v1/crates/ripgrep":
                return httpx.Response(
                    200,
                    json={"crate": {"name": "ripgrep", "max_stable_version": "15.0.0", "description": "Fast regex search"}},
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(404, text="not found")

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e47_handler)) as cli47:
            f47_ax = await safe_fetch_url("https://arxiv.org/abs/2601.00001", client=cli47, resolve_host=False, use_cache=False)
            f47_cr = await safe_fetch_url("https://crates.io/crates/ripgrep", client=cli47, resolve_host=False, use_cache=False)
        record(
            "E47",
            "正文抽取-arXiv与crates.io官方公开API结构化直取",
            f47_ax.ok and f47_ax.source_via == "arxiv-api" and f47_cr.ok and f47_cr.source_via == "crates-io-api",
            f"arxiv={f47_ax.source_via} crates={f47_cr.source_via}",
        )

        # E48: Phase 8C omni doctor 精准按引擎重置健康状态（不影响其他冷却引擎）
        mcp48 = OmniMCPServer(health_path=pathlib.Path(td) / "h48.json", env={})
        try:
            mcp48.health_store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=10**12)
            for idx in range(4):
                mcp48.health_store.record_failure("codeberg", ReasonCode.DEADLINE, now=10**12 + idx)
            doc48 = await mcp48.doctor(probe=False, reset_engines=["codeberg"])
            cb_after = mcp48.health_store.check("codeberg")
            ds_after = mcp48.health_store.check("deepseek")
        finally:
            await mcp48.aclose()
        record(
            "E48",
            "精准自愈-omni doctor按引擎重置熔断且保留其他引擎冷却态",
            doc48.get("reset_applied") == ["codeberg"]
            and cb_after.consecutive_failures == 0
            and ds_after.in_cooldown is True,
            f"reset={doc48.get('reset_applied')} ds_cooldown={ds_after.in_cooldown}",
        )

        # E49: Phase 9A research_workorders 独立模块解耦与 research re-export 兼容
        from omni_search import research as _res_mod
        from omni_search import research_workorders as _rw_mod

        record(
            "E49",
            "架构解耦-research_workorders独立模块与research零破坏re-export兼容",
            _res_mod.export_research_workorders is _rw_mod.export_research_workorders
            and _res_mod.merge_lane_results is _rw_mod.merge_lane_results,
            "re-export=identical",
        )

        # E50: Phase 9B HuggingFace 与 Wikipedia REST v1 官方 API 直取
        def _e50_handler(request: httpx.Request) -> httpx.Response:
            u = str(request.url)
            if u == "https://huggingface.co/api/models/org/demo-embed":
                return httpx.Response(
                    200,
                    json={"id": "org/demo-embed", "pipeline_tag": "feature-extraction", "downloads": 12000},
                    headers={"content-type": "application/json"},
                )
            if u == "https://en.wikipedia.org/api/rest_v1/page/summary/Hybrid_search":
                return httpx.Response(
                    200,
                    json={"title": "Hybrid search", "description": "IR technique", "extract": "Hybrid search combines lexical and dense retrieval."},
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(404, text="not found")

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e50_handler)) as cli50:
            f50_hf = await safe_fetch_url("https://huggingface.co/org/demo-embed", client=cli50, resolve_host=False, use_cache=False)
            f50_wk = await safe_fetch_url("https://en.wikipedia.org/wiki/Hybrid_search", client=cli50, resolve_host=False, use_cache=False)
        record(
            "E50",
            "正文抽取-HuggingFace与Wikipedia REST官方API结构化直取",
            f50_hf.ok and f50_hf.source_via == "huggingface-api" and f50_wk.ok and f50_wk.source_via == "wikipedia-rest-api",
            f"hf={f50_hf.source_via} wiki={f50_wk.source_via}",
        )

        # E51: Phase 9C 22 题离线 IR 黄金基准门禁
        from ir_benchmark import run_offline_ir_benchmark

        ir51 = await run_offline_ir_benchmark()
        record(
            "E51",
            "IR黄金基准-22题跨领域与对抗用例MRR与零噪声达标",
            ir51["pass"] is True and ir51["case_count"] >= 22 and ir51["metrics"]["noise_at_3"] <= 0.1,
            f"cases={ir51['case_count']} mrr={ir51['metrics']['mrr']} noise3={ir51['metrics']['noise_at_3']}",
        )

        # E52: Phase 10 v1.x 发版契约（research_audit 解耦 + 4 核心依赖 + 8 正交工具 + 版本一致）
        import tomllib
        import omni_search as _omni_pkg
        from omni_search import research_audit as _ra_mod
        from omni_search.mcp_server import _TOOLS_SPEC as _mcp_tools

        _pp = tomllib.loads((pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        _ver_tuple = tuple(int(p) for p in _omni_pkg.__version__.split(".")[:2])
        record(
            "E52",
            "发版契约-v1.x架构解耦(research_audit)+4依赖+8正交工具+版本一致",
            _omni_pkg.__version__ == _pp["version"]
            and _ver_tuple >= (1, 0)
            and len(_pp["dependencies"]) == 4
            and len(_mcp_tools) == 8
            and _res_mod.audit_freshness_and_conflicts is _ra_mod.audit_freshness_and_conflicts
            and _res_mod.classify_evidence_tier is _ra_mod.classify_evidence_tier,
            f"ver={_omni_pkg.__version__} deps={len(_pp['dependencies'])} tools={len(_mcp_tools)}",
        )

        # E53: Phase 11B DOI(Crossref/arXiv) & HackerNews 官方 API 直取 + /llms-full.txt 回退 + Markdown 协商标题抽取
        def _e53_handler(req: httpx.Request) -> httpx.Response:
            u = str(req.url)
            if u.startswith("https://api.crossref.org/works/10.1000/eval53"):
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "title": ["Eval53 Crossref Paper"],
                            "issued": {"date-parts": [[2026]]},
                            "is-referenced-by-count": 7,
                            "abstract": "<jats:p>Clean DOI abstract.</jats:p>",
                        }
                    },
                    headers={"content-type": "application/json"},
                )
            if u == "https://hacker-news.firebaseio.com/v0/item/990053.json":
                return httpx.Response(
                    200,
                    json={"id": 990053, "type": "story", "by": "alice", "title": "Eval53 HN Story", "score": 99, "descendants": 12},
                    headers={"content-type": "application/json"},
                )
            if u == "https://e53.example.com/llms.txt":
                return httpx.Response(404, text="not found")
            if u == "https://e53.example.com/llms-full.txt":
                return httpx.Response(200, text="# Full Agent Docs\n\nAll endpoints listed here.", headers={"content-type": "text/markdown"})
            if u == "https://e53.example.com/md":
                return httpx.Response(200, text="# Native MD Title\n\nBody text.", headers={"content-type": "text/markdown"})
            return httpx.Response(404, text="not found")

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e53_handler)) as _c53:
            _r53_doi = await safe_fetch_url("https://doi.org/10.1000/eval53", client=_c53, resolve_host=False, use_cache=False)
            _r53_hn = await safe_fetch_url("https://news.ycombinator.com/item?id=990053", client=_c53, resolve_host=False, use_cache=False)
            _r53_llms = await safe_fetch_url("https://e53.example.com/", client=_c53, resolve_host=False, use_cache=False, prefer_llms_txt=True)
            _r53_md = await safe_fetch_url("https://e53.example.com/md", client=_c53, resolve_host=False, use_cache=False)

        record(
            "E53",
            "安全抓取-DOI与HN官方API直取+llms-full回退+Markdown协商标题抽取",
            _r53_doi.ok and _r53_doi.source_via == "crossref-doi-api" and "Clean DOI abstract." in _r53_doi.content
            and _r53_hn.ok and _r53_hn.source_via == "hackernews-api" and "Score: 99" in _r53_hn.content
            and _r53_llms.ok and _r53_llms.source_via == "llms-full-txt"
            and _r53_md.ok and _r53_md.source_via == "markdown-negotiated" and _r53_md.title == "Native MD Title",
            f"doi={_r53_doi.source_via} hn={_r53_hn.source_via} llms={_r53_llms.source_via} md_title={_r53_md.title}",
        )

        # E54: Phase 11A/11C 全仓库源码模块千行以下健康门禁（local_query + fetch_apis 解耦）
        from omni_search import fetch_apis as _fa_mod, fetcher as _ft_mod, local_index as _li_mod, local_query as _lq_mod

        _src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "omni_search"
        _max_mod = max((len(p.read_text(encoding="utf-8").splitlines()), p.name) for p in _src_root.glob("*.py"))
        record(
            "E54",
            "架构健康-local_query与fetch_apis解耦+全模块低于1000行门禁",
            _max_mod[0] < 1000
            and _li_mod._query_db_sync is _lq_mod._query_db_sync
            and _ft_mod._try_fetch_official_api is _fa_mod._try_fetch_official_api,
            f"max_module={_max_mod[1]}:{_max_mod[0]}",
        )

        # E55: Phase 12A/12B GitHub 免配置凭据解析 + Agentic Web 响应头感知 (Content-Signal / x-markdown-tokens)
        from omni_search.contracts import resolve_github_token as _resolve_gh_tok

        _tok_gh = _resolve_gh_tok({"GH_TOKEN": "e55_gh_tok"}, allow_gh_cli=False)
        _tok_optout = _resolve_gh_tok({"OMNI_NO_GH_CLI_TOKEN": "1"}, allow_gh_cli=True)

        def _e55_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="# Agentic Header Check\n\nBody content.",
                headers={
                    "content-type": "text/markdown",
                    "content-signal": "search=yes, ai-train=no",
                    "x-markdown-tokens": "512",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e55_handler)) as _c55:
            _r55 = await safe_fetch_url("https://e55.example.com/doc", client=_c55, resolve_host=False, use_cache=False)

        record(
            "E55",
            "鉴权与协议-GH_TOKEN免配置回退+Content-Signal与x-markdown-tokens感知",
            _tok_gh == "e55_gh_tok"
            and _tok_optout == ""
            and _r55.ok
            and _r55.content_signals == {"search": "yes", "ai-train": "no"}
            and _r55.markdown_tokens == 512,
            f"signals={_r55.content_signals} md_tokens={_r55.markdown_tokens}",
        )

        # E56: Phase 13 site_map /llms-full.txt 自动发现 + omni crawl Content-Signal 限制策略告警透出
        def _e56_handler(req: httpx.Request) -> httpx.Response:
            u = str(req.url)
            if u == "https://e56.example.com/llms.txt":
                return httpx.Response(404, text="not found")
            if u == "https://e56.example.com/llms-full.txt":
                return httpx.Response(
                    200,
                    text="# Full\n- [Restricted](https://e56.example.com/restricted)\n",
                    headers={"content-type": "text/markdown"},
                )
            if u == "https://e56.example.com/restricted":
                return httpx.Response(
                    200,
                    text="# Restricted\n\nContent with restricted ai-input signal.",
                    headers={"content-type": "text/markdown", "content-signal": "ai-input=no"},
                )
            return httpx.Response(404, text="not found")

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e56_handler)) as _c56:
            _crawl56 = await safe_crawl_site(
                "https://e56.example.com",
                max_pages=2,
                client=_c56,
                resolve_host=False,
                use_cache=False,
            )

        record(
            "E56",
            "站内采集-llms-full自动发现与Content-Signal受限策略审计披露",
            _crawl56.ok
            and "llms-full-txt" in _crawl56.sources_used
            and any("content-signal-restricted" in w and "ai-input=no" in w for w in _crawl56.warnings),
            f"sources={_crawl56.sources_used} warnings={len(_crawl56.warnings)}",
        )

        # E57: Phase 14 MCP Tool annotations/title 完整 + 深路径 Markdown 误返站点首页自动回退 HTML
        from omni_search.mcp_server import _TOOLS_SPEC as _e57_tools

        _e57_ann_ok = len(_e57_tools) == 8 and all(
            isinstance(t.get("title"), str)
            and len(t["title"].strip()) >= 4
            and isinstance(t.get("annotations"), dict)
            and all(isinstance(t["annotations"].get(k), bool) for k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"))
            for t in _e57_tools
        )

        def _e57_handler(req: httpx.Request) -> httpx.Response:
            if "text/markdown" in req.headers.get("accept", ""):
                return httpx.Response(
                    200,
                    text="# Site Brand\n\n> Free AI agent.\n- [Index](https://e57.example.com/llms.txt?src=llms-txt)\n",
                    headers={"content-type": "text/markdown; charset=utf-8"},
                )
            return httpx.Response(
                200,
                text="<html><head><title>Agent Web Protocol Stack</title></head><body><article><p>Deep dive into the agent web protocol stack.</p></article></body></html>",
                headers={"content-type": "text/html; charset=utf-8"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(_e57_handler)) as _c57:
            _r57 = await safe_fetch_url(
                "https://e57.example.com/blog/agent-web-protocol-stack",
                client=_c57,
                resolve_host=False,
                use_cache=False,
            )

        record(
            "E57",
            "MCP工具annotations完整与深路径Markdown误返首页HTML回退",
            _e57_ann_ok
            and _r57.ok
            and _r57.source_via == "html-fallback-from-markdown"
            and "Protocol Stack" in _r57.title,
            f"ann_ok={_e57_ann_ok} via={_r57.source_via} title={_r57.title!r}",
        )

        # E58: Phase 15 research_planner 拆分后全模块 < 700 行 + HealthStore 凭据指纹变更自动自愈
        from omni_search.health import make_credential_fingerprint as _mk_fp

        _pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "omni_search"
        _max_lines = max(len(p.read_text(encoding="utf-8").splitlines()) for p in _pkg_dir.glob("*.py"))
        _h58_path = pathlib.Path(td) / "e58_health.json"
        _h58 = HealthStore(_h58_path)
        _h58.record_failure("github_direct", ReasonCode.FORBIDDEN, now=1000.0, cred_fingerprint="")
        _locked58 = _h58.check("github_direct", now=1010.0, cred_fingerprint="").skip_fast
        _fp58 = _mk_fp("ghp_e58_super_secret")
        _healed58 = _h58.check("github_direct", now=1020.0, cred_fingerprint=_fp58)
        _raw58 = _h58_path.read_text(encoding="utf-8") if _h58_path.exists() else ""

        record(
            "E58",
            "全模块<700行契约与凭据感知熔断自动自愈",
            _max_lines < 700
            and _locked58 is True
            and _healed58.auto_healed is True
            and _healed58.skip_fast is False
            and "ghp_e58_super_secret" not in _raw58,
            f"max_lines={_max_lines} healed={_healed58.auto_healed}",
        )

        # E59: Phase 16 CamelCase 分词召回 + local/local_code 统一 path: 子路径安全过滤
        from omni_search.local_code import _parse_local_code_query as _parse_lc59
        from omni_search.local_query import tokenize_bilingual as _tok59

        _t59 = set(_tok59("FastFailCircuitBreaker OmniMCPServer").split())
        _p59, _, _g59, _paths59 = _parse_lc59("path:src/omni_search lang:py safe_fetch_url")
        _, _, _, _bad_paths59 = _parse_lc59("path:../../etc/passwd secret")
        record(
            "E59",
            "本地分词CamelCase拆分与local_code统一path安全过滤",
            {"fastfailcircuitbreaker", "circuit", "breaker", "omnimcpserver", "mcp", "server"}.issubset(_t59)
            and _p59 == "safe_fetch_url"
            and "*.py" in _g59
            and _paths59 == ["src/omni_search"]
            and _bad_paths59 == [],
            f"toks_ok={len(_t59)} paths={_paths59}",
        )

        # E60: Phase 17 verifier 硬锚点边界防子串假阳性 + router local_code/过滤器直达
        from omni_search.verifier import _token_in_text as _tok_in60

        _d60_lc = route_domains("在本地源码里搜 FastFailCircuitBreaker", ["auto"])
        _d60_path = route_domains("path:src/omni_search circuit breaker", ["auto"])
        _d60_sym = route_domains("symbol:LocalCodeEngine", ["auto"])
        record(
            "E60",
            "引用核验硬锚点边界防幻觉与路由local_code/过滤器直达",
            _tok_in60("v2.0", "released v2.0.5") is False
            and _tok_in60("CVE-2024-1234", "patched cve-2024-12345") is False
            and _tok_in60("RAG", "memory fragmentation") is False
            and _tok_in60("1000", "indexed 1,000 docs") is True
            and "local_code" in _d60_lc
            and {"local", "local_code"}.issubset(_d60_path)
            and "local_code" in _d60_sym,
            f"lc={_d60_lc} path={_d60_path}",
        )

        # E61: Phase 18 local/local_code 统一 -path: 排除子目录过滤 + fetch_apis 模块瘦身 (<650 行)
        from omni_search.local_chunking import parse_local_query_full as _parse_full61

        _c61, _, _, _, _, _rp61, _ep61 = _parse_full61("path:src -path:tests -path:../../etc circuit")
        _, _, _g61, _ = _parse_lc59("-path:tests -path:../../etc lang:py circuit")
        _d61 = route_domains("-path:tests circuit", ["auto"])
        _max_lines61 = max(
            len(p.read_text(encoding="utf-8").splitlines())
            for p in (pathlib.Path(__file__).resolve().parent.parent / "src" / "omni_search").glob("*.py")
        )
        record(
            "E61",
            "local/local_code统一-path排除过滤与全模块低于650行守卫",
            _c61 == "circuit"
            and _rp61 == {"src"}
            and _ep61 == {"tests"}
            and "!tests/**" in _g61
            and all("etc" not in g for g in _g61)
            and {"local", "local_code"}.issubset(_d61)
            and _max_lines61 < 650,
            f"exc={sorted(_ep61)} max_lines={_max_lines61}",
        )

        # E62: Phase 19 本地知识库 after:/before: 时间过滤 + local_code symbol: 跨语言符号定义查询
        from omni_search.local_chunking import parse_local_date_bounds as _parse_dt62

        _q62, _aft62, _bef62 = _parse_dt62("after:2025-01-01 before:2026-12-31 circuit")
        _sp62, _, _sg62, _ = _parse_lc59("lang:py symbol:FastFailCircuitBreaker")
        _bad_sp62, _, _, _ = _parse_lc59("symbol:Bad.*Inject")
        record(
            "E62",
            "本地知识库after/before时间过滤与local_code symbol定义查询",
            _q62 == "circuit"
            and _aft62 is not None
            and _bef62 is not None
            and _aft62 < _bef62
            and "FastFailCircuitBreaker" in _sp62
            and "class" in _sp62
            and "*.py" in _sg62
            and ".*" not in _bad_sp62,
            f"after_ok={_aft62 is not None} sym_ok={'class' in _sp62}",
        )

        # E63: Phase 20 SSRF CGNAT 拦截 + 代码块缩进保留 + 核验极性反转拦截 + 中文长句摘录 + PyPI/Go 精准包引擎与全模块<650行
        from omni_search.fetch_security import _is_forbidden_ip as _forbid63
        from omni_search.fetcher import FetchResult as _FR63
        from omni_search.html_markdown import _normalize_extracted_text as _norm63
        from omni_search.package_engines import parse_pkg_go_dev_html as _go63
        from omni_search.research_audit import extract_relevant_quotes as _quotes63
        from omni_search.verifier import _evaluate_claim_against_page as _eval63

        _ssrf63 = _forbid63("100.100.100.200") and _forbid63("100.64.0.1") and (not _forbid63("8.8.8.8"))
        _indent63 = "    return 42" in _norm63("```python\ndef f():\n    return 42\n```")
        _pol63 = _eval63(
            "Foo does NOT support Windows v2.0",
            "https://example.com/foo",
            _FR63(ok=True, url="https://example.com/foo", title="Foo", content="Foo supports Windows v2.0 natively."),
        )
        _zh63 = _quotes63(
            "本文评测了本地混合检索在多模态文档库中的表现并对比多种策略。实验表明采用分块向量与BM25F融合的重排序架构能够显著提升中文长文档召回率并降低延迟。",
            "中文混合检索 重排序 召回率",
            max_quotes=1,
        )
        _go_items63 = _go63(
            '<div class="SearchSnippet"><a href="/github.com/spf13/cobra" data-gtmc="search result">cobra</a>'
            '<p class="SearchSnippet-synopsis">Modern Go CLI</p></div>',
            limit=2,
        )
        _max_lines63 = max(
            len(p.read_text(encoding="utf-8").splitlines())
            for p in (pathlib.Path(__file__).resolve().parent.parent / "src" / "omni_search").glob("*.py")
        )
        record(
            "E63",
            "Phase20安全SSRF/缩进/极性核验/中文摘录与Go包解析守卫",
            _ssrf63
            and _indent63
            and _pol63.verdict == "unsupported"
            and _pol63.reason_code == "polarity-conflict"
            and len(_zh63) == 1
            and len(_go_items63) == 1
            and _go_items63[0].title == "github.com/spf13/cobra"
            and _max_lines63 < 650,
            f"ssrf={_ssrf63} pol={_pol63.reason_code} zh={len(_zh63)} max_lines={_max_lines63}",
        )

    return cases


async def run_live_evals() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    srv = OmniMCPServer()

    # L01: 公网多域 L0 检索（knowledge + academic）
    r1 = await srv.pool.search("metasearch engine", domains=["knowledge", "academic"], limit=2, budget_ms=15000)
    cases.append({
        "id": "L01",
        "name": "公网-多域知识与学术检索",
        "pass": len(r1.results) >= 4 and len(r1.coverage.executed) >= 5,
        "detail": f"verdict={r1.coverage.verdict} results={len(r1.results)} exec={len(r1.coverage.executed)}",
    })

    # L02: 公网 web 域（含 SearXNG 若已配置）
    r2 = await srv.pool.search("metasearch engine", domains=["web"], limit=3, budget_ms=15000)
    cases.append({
        "id": "L02",
        "name": "公网-web 域检索与覆盖披露",
        "pass": len(r2.coverage.executed) >= 1 and len(r2.results) >= 1,
        "detail": f"verdict={r2.coverage.verdict} exec={r2.coverage.executed} results={len(r2.results)}",
    })

    # L03: 公网安全原文抓取
    f1 = await safe_fetch_url("https://en.wikipedia.org/wiki/Metasearch_engine", max_chars=800)
    cases.append({
        "id": "L03",
        "name": "公网-omni_fetch 正文抽取与边界标记",
        "pass": f1.ok and f1.status_code == 200 and f1.untrusted_content is True and len(f1.content) > 100,
        "detail": f"http={f1.status_code} len={len(f1.content)}",
    })
    return cases


async def _amain(live: bool) -> int:
    t0 = time.perf_counter()
    results = await run_offline_evals()
    if live:
        results.extend(await run_live_evals())
    passed = sum(1 for c in results if c["pass"])
    failed = len(results) - passed
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "mode": "offline+live" if live else "offline",
        "cases": results,
    }
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="omni-search 回归评测门禁")
    ap.add_argument("--live", action="store_true", help="附加公网端到端实测用例")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_amain(args.live)))


if __name__ == "__main__":
    main()
