"""Phase 20 红转绿单测：安全/流式抓取、核验极性/中文/数值精度、调研拆分、流式代码证据、包生态假活修复与遥测诊断。"""

from __future__ import annotations

import pathlib

import httpx
import pytest

from omni_search.code_evidence import GrepAppEngine, SourcegraphEngine
from omni_search.contracts import ReasonCode
from omni_search.fetcher import (
    FetchResult,
    _extract_html_title_and_body,
    _is_forbidden_ip,
    _validate_target_url,
    safe_fetch_url,
)
from omni_search.html_markdown import _normalize_extracted_text
from omni_search.mcp_server import build_default_engines
from omni_search.research_audit import audit_verified_citations, extract_relevant_quotes
from omni_search.research_planner import widen_query
from omni_search.site_map import SiteMapEntry, _score_entry
from omni_search.telemetry import TelemetryStore
from omni_search.verifier import _evaluate_claim_against_page, _extract_features


@pytest.mark.asyncio
async def test_phase20_ssrf_cgnat_fqdn_streaming_and_query_cap():
    """SSRF 阻断 100.64.0.0/10（阿里云元数据）与尾点/八进制主机；query 未命中裁剪不溢出 max_chars。"""
    import omni_search.fetch_extract as fetch_extract
    import omni_search.fetch_security as fetch_security

    assert hasattr(fetch_security, "_is_forbidden_ip")
    assert hasattr(fetch_extract, "_extract_html_title_and_body")

    assert _is_forbidden_ip("100.100.100.200") is True
    assert _is_forbidden_ip("100.64.0.1") is True
    assert _is_forbidden_ip("::ffff:100.100.100.200") is True
    assert _is_forbidden_ip("8.8.8.8") is False

    ok_fqdn, _ = _validate_target_url("http://localhost./secret", resolve_host=False)
    assert ok_fqdn is False
    ok_oct, _ = _validate_target_url("http://0177.0.0.1/admin", resolve_host=False)
    assert ok_oct is False

    long_single_para = "alpha " * 300

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "registry.npmjs.org" and req.url.path.endswith("/latest"):
            return httpx.Response(
                200,
                json={
                    "name": "demo-pkg",
                    "version": "1.2.3",
                    "description": long_single_para,
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://www.npmjs.com/package/demo-pkg",
            max_chars=300,
            query="zzzz_unmatched_query",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
    assert res.ok is True
    assert res.source_via == "npm-registry-api"
    assert len(res.content) <= 300
    assert res.truncated is True


@pytest.mark.asyncio
async def test_phase20_markdown_code_indent_article_header_npm_latest_and_sitemap():
    """保留代码块缩进与 <article><header> 主标题；npm 优先 /latest；site_map llms-full-txt 同权。"""
    html = (
        "<html><head><title>Doc</title></head><body>"
        "<header><nav>Global Nav</nav></header>"
        "<article>"
        "<header><h1>Article Main Title</h1></header>"
        "<p>Intro paragraph about indentation.</p>"
        '<pre><code class="language-python">'
        "def compute(x):\n"
        "    if x > 0:\n"
        "        return x * 2\n"
        "    return 0\n"
        "</code></pre>"
        "</article></body></html>"
    )
    _, body = _extract_html_title_and_body(html)
    assert "Global Nav" not in body
    assert "# Article Main Title" in body
    assert "    if x > 0:" in body
    assert "        return x * 2" in body

    raw_md = "Intro   text\n\n```yaml\nserver:\n  port: 8080\n    nested: true\n```\n\nFooter   line"
    norm = _normalize_extracted_text(raw_md)
    assert "  port: 8080" in norm
    assert "    nested: true" in norm
    assert "Intro text" in norm

    seen_urls: list[str] = []

    def npm_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        if str(req.url) == "https://registry.npmjs.org/zod/latest":
            return httpx.Response(
                200,
                json={
                    "name": "zod",
                    "version": "4.6.7",
                    "description": "TypeScript-first schema validation.",
                    "homepage": "https://zod.dev",
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(npm_handler)) as cli:
        nres = await safe_fetch_url("https://www.npmjs.com/package/zod", client=cli, resolve_host=False, use_cache=False)
    assert nres.ok is True
    assert nres.source_via == "npm-registry-api"
    assert seen_urls == ["https://registry.npmjs.org/zod/latest"]
    assert "4.6.7" in nres.content

    s_llms = _score_entry(SiteMapEntry(url="https://docs.example.com/guide", title="Guide", source="llms-txt"), [])
    s_full = _score_entry(SiteMapEntry(url="https://docs.example.com/guide", title="Guide", source="llms-full-txt"), [])
    s_map = _score_entry(SiteMapEntry(url="https://docs.example.com/guide", title="Guide", source="sitemap"), [])
    assert s_full == s_llms
    assert s_full > s_map


def test_phase20_verifier_polarity_cjk_sliding_and_unit_anchors():
    """verifier 检测否定反转冲突、中文奇数偏移稳定召回、带单位数值纳入硬锚点。"""
    pos_page = FetchResult(
        ok=True,
        url="https://example.com/foo",
        title="Foo Release Notes",
        content="Foo supports Windows v2.0 natively with full async IO.",
        untrusted_content=True,
    )
    neg_claim_res = _evaluate_claim_against_page("Foo does NOT support Windows v2.0", pos_page.url, pos_page)
    assert neg_claim_res.verdict == "unsupported"
    assert neg_claim_res.reason_code == "polarity-conflict"

    neg_page = FetchResult(
        ok=True,
        url="https://example.com/foo-neg",
        title="Foo Release Notes",
        content="Foo does not support Windows v2.0 in this release.",
        untrusted_content=True,
    )
    pos_claim_res = _evaluate_claim_against_page("Foo supports Windows v2.0 natively", neg_page.url, neg_page)
    assert pos_claim_res.verdict == "unsupported"
    assert pos_claim_res.reason_code == "polarity-conflict"

    # 页面其他无关段落含 not 不应误杀正常肯定主张
    mixed_page = FetchResult(
        ok=True,
        url="https://example.com/mixed",
        title="Foo Docs",
        content="Foo supports Windows v2.0 natively.\nLegacy plugins are not enabled by default.",
        untrusted_content=True,
    )
    ok_res = _evaluate_claim_against_page("Foo supports Windows v2.0 natively", mixed_page.url, mixed_page)
    assert ok_res.verdict == "supported"

    # 中文奇数偏移稳定支持
    zh_page = FetchResult(
        ok=True,
        url="https://example.com/zh",
        title="深度学习框架报告",
        content="实测表明深度学习框架性能提升显著，吞吐翻倍。",
        untrusted_content=True,
    )
    zh_res = _evaluate_claim_against_page("的深度学习框架性能提升", zh_page.url, zh_page)
    assert zh_res.verdict == "supported"

    # 带单位数值纳入硬锚点并防止编造数字
    anchors, _ = _extract_features("延迟仅 12ms，内存占用 64MB，加速 3x")
    low_anchors = {a.lower().replace(" ", "") for a in anchors}
    assert {"12ms", "64mb", "3x"}.issubset(low_anchors)

    perf_page = FetchResult(
        ok=True,
        url="https://example.com/perf",
        title="Benchmark",
        content="实测服务延迟 99ms，内存占用 512MB，加速 1x。",
        untrusted_content=True,
    )
    fake_num_res = _evaluate_claim_against_page("实测服务延迟仅 12ms，内存占用 64MB，加速 3x", perf_page.url, perf_page)
    assert fake_num_res.verdict != "supported"


def test_phase20_research_split_cjk_quotes_and_widen():
    """research_models/research_evidence 独立拆分；中文无空格段落可摘录；宽词不劈碎专名；核验保留标题。"""
    import omni_search.research_evidence as research_evidence
    import omni_search.research_models as research_models

    assert hasattr(research_models, "ResearchReport")
    assert hasattr(research_evidence, "_merge_lane_candidates")

    # research.py 不应再包含中尾部 E402 环形导入
    src_research = (
        pathlib.Path(__file__).resolve().parent.parent / "src" / "omni_search" / "research.py"
    ).read_text(encoding="utf-8")
    assert "# noqa: E402" not in src_research

    # 中文无空格长段落（>80字）应按中文句号分句并提取相关引文
    zh_doc = (
        "本文系统评测了本地混合检索引擎在多模态文档库中的表现并对比了多种融合策略与向量压缩方案以及内存占用开销。"
        "实验表明采用分块向量与BM25F融合的重排序架构能够在毫秒级延迟内将中文长文档召回率提升至百分之九十五以上并显著降低误报率。"
    )
    quotes = extract_relevant_quotes(zh_doc, "中文混合检索 重排序 召回率", max_quotes=2)
    assert len(quotes) >= 1
    assert any("重排序" in q for q in quotes)

    # audit_verified_citations 应保留 ev.title
    ev = research_models.VerifiedEvidence(
        url="https://example.com/sqlite",
        title="SQLite FTS5 Hybrid Retrieval Architecture",
        evidence_tier="T1",
        verified=True,
        status_code=200,
        excerpt="Combines lexical BM25 ranking with vector embeddings.",
    )
    aud = audit_verified_citations("SQLite FTS5 Hybrid Retrieval", [ev])
    assert aud is not None
    assert aud.supported_count == 1

    # widen_query 不应因“中/为”单字切碎“中间件”与“华为”
    widened = widen_query("分布式中间件架构设计与华为鸿蒙内核", max_terms=4)
    assert "中间件" in widened
    assert "华为" in widened
    assert "间件" not in widened.replace("中间件", "")


@pytest.mark.asyncio
async def test_phase20_streaming_code_evidence_pkg_engines_and_telemetry(tmp_path: pathlib.Path):
    """Sourcegraph 流式 chunkMatches、grep.app 语言别名、PyPI/Go 真包检索引擎与遥测假活诊断。"""
    # 1) Sourcegraph V3 chunkMatches 流式解析
    sse_body = (
        ": heartbeat\n\n"
        "id: 1\n"
        "event: matches\n"
        'data: [{"type":"content","repository":"github.com/org/proj","path":"src/app.py",'
        '"chunkMatches":[{"content":"def fuse_results():\\n    return []","contentStart":{"line":41}}]}]\n\n'
        "event: done\n"
        "data: {}\n\n"
    )
    sg = SourcegraphEngine(env={})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"}))
    ) as cli:
        sg_env = await sg.search("fuse_results", limit=1, client=cli)
    assert sg_env.failure is None
    assert len(sg_env.results) == 1
    assert sg_env.results[0].extra["line"] == 42
    assert "def fuse_results" in sg_env.results[0].snippet

    # 2) GrepApp 语言别名规范化与 403 -> FORBIDDEN
    captured_lang: list[object] = []

    def grep_handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(req.content.decode("utf-8"))
        args = body["params"]["arguments"]
        captured_lang.append(args.get("language"))
        if args.get("query") == "forbidden_case":
            return httpx.Response(403, text="Forbidden")
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": ""}]}},
        )

    ga = GrepAppEngine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(grep_handler)) as cli:
        await ga.search("lang:py asyncio.gather", limit=3, client=cli)
        f_env = await ga.search("forbidden_case", limit=3, client=cli)
    assert captured_lang[0] == ["Python"]
    assert f_env.failure is not None
    assert f_env.failure.reason_code == ReasonCode.FORBIDDEN

    # 3) PyPI / Go 原生包检索引擎按 query 精准返回，且默认总引擎数保持 50
    engines = build_default_engines(env={})
    assert len(engines) == 50
    pypi_eng = next(e for e in engines if e.id == "pypi_ecosystems")
    go_eng = next(e for e in engines if e.id == "golang_ecosystems")

    def pkg_handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if "package_names" in u and "prefix=fastapi" in u:
            return httpx.Response(200, json=["fastapi", "fastapi-utils"])
        if u == "https://pypi.org/pypi/fastapi/json":
            return httpx.Response(
                200,
                json={
                    "info": {
                        "name": "fastapi",
                        "version": "0.115.0",
                        "summary": "FastAPI framework, high performance",
                        "project_url": "https://pypi.org/project/fastapi/",
                    }
                },
            )
        if u == "https://pypi.org/pypi/fastapi-utils/json":
            return httpx.Response(
                200,
                json={
                    "info": {
                        "name": "fastapi-utils",
                        "version": "0.8.0",
                        "summary": "Reusable utilities for FastAPI",
                        "project_url": "https://pypi.org/project/fastapi-utils/",
                    }
                },
            )
        if req.url.host == "pkg.go.dev":
            html = (
                '<div class="SearchSnippet">'
                '<a href="/github.com/spf13/cobra" data-gtmc="search result">cobra</a>'
                '<p class="SearchSnippet-synopsis">A Commander for modern Go CLI interactions.</p>'
                "</div>"
            )
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(pkg_handler)) as cli:
        p_env = await pypi_eng.search("fastapi", limit=2, client=cli)
        g_env = await go_eng.search("cobra cli", limit=2, client=cli)
    assert p_env.failure is None
    assert len(p_env.results) == 2
    assert p_env.results[0].title.lower().startswith("fastapi")
    assert all("requests" != r.title.lower() for r in p_env.results)
    assert g_env.failure is None
    assert len(g_env.results) == 1
    assert g_env.results[0].url == "https://pkg.go.dev/github.com/spf13/cobra"

    # 4) TelemetryStore 剥离 missing-credential 且检测高产出零贡献（假活源）
    tstore = TelemetryStore(tmp_path / "tel.db")
    events = []
    for _ in range(6):
        events.append(
            {
                "engine": "brave",
                "domain": "web",
                "ok": False,
                "reason_code": "missing-credential",
                "elapsed_ms": 0,
                "result_count": 0,
                "top5_hits": 0,
                "skipped_fast": False,
            }
        )
        events.append(
            {
                "engine": "ghost_pkg",
                "domain": "packages",
                "ok": True,
                "reason_code": "ok",
                "elapsed_ms": 900,
                "result_count": 5,
                "top5_hits": 0,
                "skipped_fast": False,
            }
        )
    tstore.record_batch(events)
    summary = tstore.summarize(window_days=7)
    by_eng = {e["engine"]: e for e in summary["engines"]}
    assert by_eng["brave"]["live_calls"] == 0
    assert not any("[brave]" in rec for rec in summary["recommendations"])
    assert any("[ghost_pkg]" in rec for rec in summary["recommendations"])
