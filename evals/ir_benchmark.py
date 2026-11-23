"""Search All in One 信息检索（IR）黄金评测基准：量化 MRR / Hit@3 / Noise@3 / 延迟与 Token 压缩比。"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import struct
import sys
import tempfile
import time
from typing import Any

from omni_search.contracts import (
    CompletenessReport,
    EngineEnvelope,
    LedgerSummary,
    SearchItem,
    SearchResponse,
    compact_search_response,
    build_coverage,
)
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.mcp_server import OmniMCPServer
from omni_search.pool import fuse_results_rrf
from omni_search.research import _merge_lane_candidates, _select_diverse_verify_targets


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return round(s[idx], 2)


def _make_png(w: int, h: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


async def run_offline_ir_benchmark() -> dict[str, Any]:
    """离线确定性 14 条黄金查询 IR 质量基准（本地双语/代码/多模态/集合隔离 + 跨源相关性与 T1 抗噪）。"""
    t0 = time.perf_counter()
    cases_out: list[dict[str, Any]] = []
    latencies: list[float] = []
    mrr_scores: list[float] = []
    hit3_flags: list[int] = []
    noise3_rates: list[float] = []

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        work = root / "work_kb"
        notes = root / "notes_kb"
        work.mkdir()
        notes.mkdir()

        # 构造中英双语 + 代码 + CSV + 图像 + Frontmatter 黄金语料库
        (work / "sqlite_hybrid_rag.md").write_text(
            "---\ntitle: SQLite Hybrid RAG\ntags: [rag, sqlite, fts5]\n---\n\n"
            "# SQLite FTS5 + Vector Hybrid Search\n\n"
            "Combines BM25F lexical search with chunk-level embeddings and Reciprocal Rank Fusion (RRF).\n",
            encoding="utf-8",
        )
        (work / "zh_memory_arch.md").write_text(
            "---\ntitle: 智能体记忆架构\ntags: [agent, memory]\n---\n\n"
            "# 大模型智能体长期记忆混合检索\n\n"
            "本文总结了向量检索与稀疏关键词重排序在智能体工作记忆中的协同机制。\n",
            encoding="utf-8",
        )
        (work / "circuit_breaker.py").write_text(
            "class FastFailCircuitBreaker:\n"
            "    def should_skip_quota_exhausted(self, fails: int) -> bool:\n"
            "        return fails >= 1\n",
            encoding="utf-8",
        )
        (work / "metrics_report.csv").write_text(
            "engine_id,domain,p95_latency_ms\n"
            + "\n".join(f"eng_{i},web,{100 + i}" for i in range(40))
            + "\nsearxng_cluster,web,42\n",
            encoding="utf-8",
        )
        (work / "ui_mockup_4k.png").write_bytes(_make_png(3840, 2160))
        (work / "mobile_portrait.png").write_bytes(_make_png(1080, 1920))
        (notes / "cooking_recipes.md").write_text(
            "---\ntitle: Pasta Guide\ntags: [cooking]\n---\n\n"
            "Boil water, add olive oil and fresh basil for classic tomato pasta.\n",
            encoding="utf-8",
        )
        (notes / "personal_rag_diary.md").write_text(
            "---\ntitle: Personal RAG Notes\ntags: [diary, rag]\n---\n\n"
            "Personal thoughts on hybrid search indexing over local markdown journals.\n",
            encoding="utf-8",
        )

        db = root / "ir_golden.db"
        index_paths([work], db_path=db, collection="work")
        index_paths([notes], db_path=db, collection="notes")
        eng = LocalIndexEngine(db_path=db)

        local_golden: list[tuple[str, str, set[str], set[str]]] = [
            ("IR01", "sqlite fts5 reciprocal rank fusion", {"sqlite_hybrid_rag.md"}, {"cooking_recipes.md"}),
            ("IR02", "大模型智能体长期记忆混合检索", {"zh_memory_arch.md"}, {"cooking_recipes.md"}),
            ("IR03", "type:code FastFailCircuitBreaker", {"circuit_breaker.py"}, {"sqlite_hybrid_rag.md"}),
            ("IR04", "collection:work rag", {"sqlite_hybrid_rag.md"}, {"personal_rag_diary.md"}),
            ("IR05", "collection:notes tag:rag", {"personal_rag_diary.md"}, {"sqlite_hybrid_rag.md"}),
            ("IR06", "type:image 4k", {"ui_mockup_4k.png"}, {"mobile_portrait.png"}),
            ("IR07", "type:image portrait", {"mobile_portrait.png"}, {"ui_mockup_4k.png"}),
            ("IR08", "ext:.csv searxng_cluster", {"metrics_report.csv"}, {"cooking_recipes.md"}),
            ("IR09", "tag:memory 智能体", {"zh_memory_arch.md"}, {"cooking_recipes.md"}),
            ("IR10", "tomato basil pasta", {"cooking_recipes.md"}, {"sqlite_hybrid_rag.md"}),
            ("IR23", "type:code circuit breaker", {"circuit_breaker.py"}, {"sqlite_hybrid_rag.md"}),
            ("IR24", "path:work_kb rag", {"sqlite_hybrid_rag.md"}, {"personal_rag_diary.md"}),
        ]

        for cid, q, expected, forbidden in local_golden:
            q0 = time.perf_counter()
            env = await eng.search(q, limit=5)
            lat = (time.perf_counter() - q0) * 1000.0
            latencies.append(lat)
            top3 = [r.title for r in env.results[:3]]
            rr = 0.0
            for idx, t in enumerate(top3, start=1):
                if t in expected:
                    rr = 1.0 / idx
                    break
            hit3 = 1 if any(t in expected for t in top3) else 0
            noise3 = sum(1 for t in top3 if t in forbidden) / max(1, len(top3))
            mrr_scores.append(rr)
            hit3_flags.append(hit3)
            noise3_rates.append(noise3)
            cases_out.append(
                {"id": cid, "kind": "local", "query": q, "top3": top3, "rr": round(rr, 3), "noise3": round(noise3, 3), "ms": round(lat, 2)}
            )

        # 4 条跨源融合 & T1 泛词抗噪对抗用例（验证 pool.py + research.py 排序质量）
        def _mk_resp(q: str, items: list[SearchItem]) -> SearchResponse:
            return SearchResponse(
                version="0.1.0",
                query=q,
                domains=["web", "knowledge", "code"],
                results=items,
                coverage=build_coverage([]),
                completeness=CompletenessReport(returned_count=len(items), total_candidates=len(items), truncated=False, per_source_limit=5),
                ledger=LedgerSummary(run_id="ir", elapsed_ms=10, budget_ms=1000),
            )

        adversarial_cases: list[tuple[str, str, list[SearchItem], set[str], set[str]]] = [
            (
                "IR11",
                "AI agent web search MCP server markdown extraction reranking",
                [
                    SearchItem(id="n1", domain="knowledge", source="mdn", title="Web server - Glossary", url="https://developer.mozilla.org/en-US/docs/Glossary/Web_server", snippet="A web server is software serving HTTP.", score=0.03),
                    SearchItem(id="g1", domain="code", source="github_direct", title="unclecode/crawl4ai - MCP markdown extraction", url="https://github.com/unclecode/crawl4ai", snippet="Open-source web crawler for AI agents with clean markdown extraction and reranking.", score=0.018),
                    SearchItem(id="g2", domain="academic", source="arxiv", title="MCP-Universe: Benchmarking MCP Servers", url="https://arxiv.org/abs/2508.14704", snippet="Evaluation benchmark for Model Context Protocol MCP servers.", score=0.016),
                ],
                {"https://github.com/unclecode/crawl4ai", "https://arxiv.org/abs/2508.14704"},
                {"https://developer.mozilla.org/en-US/docs/Glossary/Web_server"},
            ),
            (
                "IR12",
                "local hybrid search sqlite vector fts5 markitdown docling",
                [
                    SearchItem(id="n2", domain="knowledge", source="mdn", title="search.search()", url="https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/search/search", snippet="Perform a search using browser search engine.", score=0.03),
                    SearchItem(id="n3", domain="academic", source="crossref", title="VLSI Circuit Partitioning Genetic Local Search", url="https://doi.org/10.1109/vlsi.2004.1", snippet="Genetic local search for VLSI circuits.", score=0.025),
                    SearchItem(id="g3", domain="code", source="github_direct", title="DocMemory SQLite FTS5 Hybrid Search", url="https://github.com/kuchris/DocMemory", snippet="Local hybrid search with SQLite FTS5 vector embeddings and markitdown.", score=0.017),
                ],
                {"https://github.com/kuchris/DocMemory"},
                {"https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/search/search", "https://doi.org/10.1109/vlsi.2004.1"},
            ),
            (
                "IR13",
                "CVE-2026-12345 openssl buffer overflow",
                [
                    SearchItem(id="n4", domain="knowledge", source="wikipedia", title="Buffer (computer science)", url="https://en.wikipedia.org/wiki/Data_buffer", snippet="General computer science definition of buffer.", score=0.02),
                    SearchItem(id="g4", domain="security", source="nvd_cve", title="CVE-2026-12345 OpenSSL Buffer Overflow", url="https://nvd.nist.gov/vuln/detail/CVE-2026-12345", snippet="Critical buffer overflow vulnerability in OpenSSL.", score=0.019),
                ],
                {"https://nvd.nist.gov/vuln/detail/CVE-2026-12345"},
                {"https://en.wikipedia.org/wiki/Data_buffer"},
            ),
            (
                "IR14",
                "Google News redirect demotion check",
                [
                    SearchItem(id="gn", domain="news", source="google_news_rss", title="Google News redirect check", url="https://news.google.com/rss/articles/CBMiFAKE", snippet="Redirect article", score=0.9, extra={"evidence_tier": "T3"}),
                    SearchItem(id="dr", domain="news", source="hackernews", title="Direct primary article on redirect check", url="https://blog.example.org/direct-article", snippet="Direct technical post", score=0.3, extra={"evidence_tier": "T3"}),
                ],
                {"https://blog.example.org/direct-article"},
                {"https://news.google.com/rss/articles/CBMiFAKE"},
            ),
            (
                "IR15",
                "MCP Streamable HTTP SSE progress notification spec",
                [
                    SearchItem(id="n15", domain="code", source="github_issues", title="[AUTOCUT] Distribution Build Failed for sql-2.10.0", url="https://github.com/opensearch-project/sql/issues/2026", snippet="Build failed for sql 2.10.0 notification log.", score=0.04),
                    SearchItem(id="g15", domain="knowledge", source="mdn", title="MCP Streamable HTTP Transport with SSE Progress", url="https://modelcontextprotocol.io/specification/streamable-http", snippet="Streamable HTTP transport supports SSE notifications/progress events for MCP servers.", score=0.02),
                ],
                {"https://modelcontextprotocol.io/specification/streamable-http"},
                {"https://github.com/opensearch-project/sql/issues/2026"},
            ),
            (
                "IR16",
                "中文混合检索 BM25F RRF 重排序 精度评测",
                [
                    SearchItem(id="n16", domain="knowledge", source="wikipedia", title="排序算法列表", url="https://zh.wikipedia.org/wiki/排序算法", snippet="快速排序与归并排序的时间复杂度介绍。", score=0.03),
                    SearchItem(id="g16", domain="academic", source="arxiv", title="中文混合检索 BM25F 与 RRF 融合重排序精度评测", url="https://arxiv.org/abs/2605.01234", snippet="针对中文文档构建 BM25F 与稠密向量 RRF 融合重排序基准。", score=0.02),
                ],
                {"https://arxiv.org/abs/2605.01234"},
                {"https://zh.wikipedia.org/wiki/排序算法"},
            ),
            (
                "IR17",
                "deep research citation hallucination verification benchmark",
                [
                    SearchItem(id="n17", domain="news", source="gdelt", title="Research Park Opens New Building", url="https://localnews.example.com/park", snippet="A new research building opened downtown.", score=0.03),
                    SearchItem(id="g17", domain="academic", source="openalex", title="Detecting Reference Hallucinations in Deep Research Agents", url="https://doi.org/10.48550/arxiv.2604.03173", snippet="Systematic citation verification benchmark measuring reference hallucinations in deep research agents.", score=0.02),
                ],
                {"https://doi.org/10.48550/arxiv.2604.03173"},
                {"https://localnews.example.com/park"},
            ),
            (
                "IR18",
                "OSV vulnerability advisory GHSA-xxxx python pypdf",
                [
                    SearchItem(id="n18", domain="packages", source="pypi_ecosystems", title="python-dateutil", url="https://pypi.org/project/python-dateutil/", snippet="Extensions to the standard Python datetime module.", score=0.03),
                    SearchItem(id="g18", domain="security", source="osv_vulns", title="OSV Advisory for pypdf vulnerability", url="https://osv.dev/vulnerability/PYSEC-2026-0001", snippet="Security advisory in Python pypdf PDF parser.", score=0.02),
                ],
                {"https://osv.dev/vulnerability/PYSEC-2026-0001"},
                {"https://pypi.org/project/python-dateutil/"},
            ),
            (
                "IR19",
                "rust crates.io serde json deserialization zero copy",
                [
                    SearchItem(id="n19", domain="news", source="gdelt", title="Rust Belt Manufacturing Jobs Rise", url="https://news.example.com/rust-belt", snippet="Manufacturing output increased this quarter.", score=0.03),
                    SearchItem(id="g19", domain="packages", source="crates_io", title="serde - Rust serialization and zero-copy deserialization framework", url="https://crates.io/crates/serde", snippet="A generic serialization/deserialization framework supporting JSON and zero-copy parsing in Rust.", score=0.02),
                ],
                {"https://crates.io/crates/serde"},
                {"https://news.example.com/rust-belt"},
            ),
            (
                "IR20",
                "huggingface model2vec static embeddings multilingual",
                [
                    SearchItem(id="n20", domain="knowledge", source="wikipedia", title="Vector space", url="https://en.wikipedia.org/wiki/Vector_space", snippet="A vector space is a mathematical set of elements called vectors.", score=0.03),
                    SearchItem(id="g20", domain="media", source="hf_models", title="minishlab/potion-multilingual-128M static embeddings", url="https://huggingface.co/minishlab/potion-multilingual-128M", snippet="Fast CPU multilingual static embeddings trained with Model2Vec on HuggingFace.", score=0.02),
                ],
                {"https://huggingface.co/minishlab/potion-multilingual-128M"},
                {"https://en.wikipedia.org/wiki/Vector_space"},
            ),
            (
                "IR21",
                "arXiv 2508.14704 hybrid retrieval sqlite fts5",
                [
                    SearchItem(id="n21", domain="knowledge", source="mdn", title="IndexedDB API tutorial", url="https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API", snippet="IndexedDB is a low-level API for client-side storage.", score=0.03),
                    SearchItem(id="g21", domain="academic", source="arxiv", title="Hybrid Retrieval with SQLite FTS5 and Dense Vectors (arXiv:2508.14704)", url="https://arxiv.org/abs/2508.14704", snippet="Local hybrid retrieval architecture using SQLite FTS5 and reciprocal rank fusion.", score=0.02),
                ],
                {"https://arxiv.org/abs/2508.14704"},
                {"https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API"},
            ),
            (
                "IR22",
                "PyPI httpx async HTTP/2 client timeout retry",
                [
                    SearchItem(id="n22", domain="web", source="ddg_instant", title="Python Beginner Tutorial 2026", url="https://blog.example.org/python-basics", snippet="Learn variables, loops, and functions in Python.", score=0.03),
                    SearchItem(id="g22", domain="packages", source="pypi_ecosystems", title="httpx - Async HTTP/2 client for Python", url="https://pypi.org/project/httpx/", snippet="Next generation async HTTP/2 client with timeout and retry support on PyPI.", score=0.02),
                ],
                {"https://pypi.org/project/httpx/"},
                {"https://blog.example.org/python-basics"},
            ),
        ]

        for cid, brief, items, exp_urls, forb_urls in adversarial_cases:
            q0 = time.perf_counter()
            if cid == "IR14":
                ranked = _select_diverse_verify_targets(items, top_k=1)
            else:
                merged = _merge_lane_candidates([("l1", _mk_resp(brief, items))], brief=brief)
                ranked = _select_diverse_verify_targets(merged, top_k=2)
            lat = (time.perf_counter() - q0) * 1000.0
            latencies.append(lat)
            top_urls = [c.url for c in ranked[:2]]
            rr = 0.0
            for idx, u in enumerate(top_urls, start=1):
                if u in exp_urls:
                    rr = 1.0 / idx
                    break
            hit3 = 1 if any(u in exp_urls for u in top_urls) else 0
            # 对于对抗用例，干扰项不得进入第 1 名核验
            noise3 = 1.0 if (top_urls and top_urls[0] in forb_urls) else 0.0
            mrr_scores.append(rr)
            hit3_flags.append(hit3)
            noise3_rates.append(noise3)
            cases_out.append(
                {"id": cid, "kind": "adversarial", "query": brief, "top_urls": top_urls, "rr": round(rr, 3), "noise": round(noise3, 3), "ms": round(lat, 2)}
            )

        # 计算 concise 省 Token 压缩比
        sample_items = [
            SearchItem(
                id=f"item-{i}",
                domain="code",
                source="github_direct",
                title=f"org/repo-{i}",
                url=f"https://github.com/org/repo-{i}",
                snippet="Detailed repository description " * 12,
                published_at="2026-10-01T00:00:00Z",
                popularity=1200 + i,
                unit="stars",
                score=0.05 - i * 0.001,
                extra={"matched_sources": ["github_direct", "searxng"], "evidence_tier": "T1", "raw_blob": "x" * 120},
            )
            for i in range(10)
        ]
        full_resp = _mk_resp("agent search benchmark", sample_items)
        full_len = len(full_resp.model_dump_json())
        concise_len = len(json.dumps(compact_search_response(full_resp), ensure_ascii=False))
        saving_ratio = round(1.0 - (concise_len / max(1, full_len)), 3)

    mrr = round(sum(mrr_scores) / max(1, len(mrr_scores)), 4)
    hit_at_3 = round(sum(hit3_flags) / max(1, len(hit3_flags)), 4)
    noise_at_3 = round(sum(noise3_rates) / max(1, len(noise3_rates)), 4)
    passed = (mrr >= 0.85) and (hit_at_3 >= 0.90) and (noise_at_3 <= 0.10) and (saving_ratio >= 0.30)

    return {
        "mode": "offline",
        "case_count": len(cases_out),
        "pass": passed,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "thresholds": {"mrr_min": 0.85, "hit_at_3_min": 0.90, "noise_at_3_max": 0.10, "concise_saving_min": 0.30},
        "metrics": {
            "mrr": mrr,
            "hit_at_3": hit_at_3,
            "noise_at_3": noise_at_3,
            "p50_ms": _pct(latencies, 0.50),
            "p95_ms": _pct(latencies, 0.95),
            "concise_saving_ratio": saving_ratio,
        },
        "cases": cases_out,
    }


async def run_live_ir_benchmark() -> dict[str, Any]:
    """可选公网 5 条黄金查询端到端指标采集（延迟 P50/P95、命中数、覆盖率与 Token 压缩比）。"""
    srv = OmniMCPServer()
    queries = [
        ("L_IR01", "sqlite-vec hybrid search", ["code", "packages"]),
        ("L_IR02", "Model Context Protocol specification", ["knowledge", "academic"]),
        ("L_IR03", "CVE-2024-3094 xz", ["security", "news"]),
        ("L_IR04", "大模型混合检索重排序", ["knowledge", "academic"]),
        ("L_IR05", "crawl4ai markdown", ["web", "code"]),
    ]
    rows: list[dict[str, Any]] = []
    lats: list[float] = []
    savings: list[float] = []
    try:
        for cid, q, doms in queries:
            resp = await srv.pool.search(q, domains=doms, limit=4, budget_ms=15000, use_cache=False)
            lats.append(float(resp.ledger.elapsed_ms))
            fl = len(resp.model_dump_json())
            cl = len(json.dumps(compact_search_response(resp), ensure_ascii=False))
            sav = round(1.0 - cl / max(1, fl), 3)
            savings.append(sav)
            rows.append(
                {
                    "id": cid,
                    "query": q,
                    "domains": doms,
                    "verdict": resp.coverage.verdict,
                    "results": len(resp.results),
                    "executed": len(resp.coverage.executed),
                    "elapsed_ms": resp.ledger.elapsed_ms,
                    "concise_saving_ratio": sav,
                }
            )
    finally:
        await srv.aclose()

    return {
        "mode": "live",
        "case_count": len(rows),
        "pass": all(r["results"] >= 1 for r in rows),
        "metrics": {
            "p50_ms": _pct(lats, 0.50),
            "p95_ms": _pct(lats, 0.95),
            "avg_results": round(sum(r["results"] for r in rows) / max(1, len(rows)), 2),
            "avg_concise_saving_ratio": round(sum(savings) / max(1, len(savings)), 3),
        },
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Search All in One IR 质量黄金评测基准")
    ap.add_argument("--live", action="store_true", help="附加公网 5 条跨领域查询端到端实测")
    args = ap.parse_args()

    async def _run() -> int:
        offline = await run_offline_ir_benchmark()
        out: dict[str, Any] = {"offline": offline}
        ok = bool(offline["pass"])
        if args.live:
            live = await run_live_ir_benchmark()
            out["live"] = live
            ok = ok and bool(live["pass"])
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
