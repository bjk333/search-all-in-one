import json
import pathlib

import httpx
import pytest

from omni_search.contracts import ReasonCode
from omni_search.declarative import DeclarativeEngine, load_declarative_engines
from omni_search.router import route_domains


@pytest.mark.asyncio
async def test_all_9_new_native_code_and_package_yaml_specs_load_and_parse():
    """验证 9 个新增原生代码托管与包生态 YAML（gitlab/codeberg/gitee/maven/nuget/rubygems/packagist/dockerhub/mcp_registry）。"""
    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = {s.id: s for s in load_declarative_engines(engines_dir)}
    expected = {
        "gitlab",
        "codeberg",
        "gitee",
        "maven",
        "nuget",
        "rubygems",
        "packagist",
        "dockerhub",
        "mcp_registry",
    }
    assert expected.issubset(specs.keys())

    # Maven url_template 拼接
    maven_eng = DeclarativeEngine(specs["maven"], env={})
    maven_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"response": {"docs": [{"id": "org.xerial:sqlite-jdbc", "g": "org.xerial", "a": "sqlite-jdbc", "latestVersion": "3.47.0"}]}},
        )
    )
    async with httpx.AsyncClient(transport=maven_mock) as cli:
        r_m = await maven_eng.search("sqlite", limit=2, client=cli)
    assert r_m.failure is None
    assert r_m.results[0].url == "https://search.maven.org/artifact/org.xerial/sqlite-jdbc"

    # RubyGems 根数组解析
    rg_eng = DeclarativeEngine(specs["rubygems"], env={})
    rg_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json=[{"name": "sqlite3", "version": "2.1.0", "info": "SQLite3 Ruby binding", "project_uri": "https://rubygems.org/gems/sqlite3", "downloads": 99999}],
        )
    )
    async with httpx.AsyncClient(transport=rg_mock) as cli:
        r_g = await rg_eng.search("sqlite", limit=2, client=cli)
    assert r_g.failure is None
    assert r_g.results[0].title == "sqlite3"
    assert r_g.results[0].popularity == 99999

    # MCP Registry 嵌套 server 字段
    mcp_eng = DeclarativeEngine(specs["mcp_registry"], env={})
    mcp_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"servers": [{"server": {"name": "io.demo/omni-search", "description": "Universal search MCP", "websiteUrl": "https://example.com/omni"}}]},
        )
    )
    async with httpx.AsyncClient(transport=mcp_mock) as cli:
        r_mcp = await mcp_eng.search("search", limit=2, client=cli)
    assert r_mcp.failure is None
    assert r_mcp.results[0].title == "io.demo/omni-search"
    assert r_mcp.results[0].url == "https://example.com/omni"


@pytest.mark.asyncio
async def test_grep_app_engine_parses_sse_and_detects_200_rate_limit():
    """GrepAppEngine 应解析官方 MCP SSE 代码片段，并将 HTTP 200+isError 限流归类为 RATE_LIMITED。"""
    from omni_search.code_evidence import GrepAppEngine

    sample_text = (
        "Repository: bytedance/deer-flow\n"
        "Path: backend/retrieval.py\n"
        "URL: https://github.com/bytedance/deer-flow/blob/main/backend/retrieval.py\n"
        "--- Snippet 1 (Line 42) ---\n"
        "def fuse_rrf(scores):\n    return sorted(scores)\n"
    )
    sse_ok = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": sample_text}], "isError": False}}) + "\n\n"

    eng = GrepAppEngine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=sse_ok))) as cli:
        env = await eng.search("fuse_rrf", limit=3, client=cli)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].domain == "code_evidence"
    assert env.results[0].extra["file"] == "backend/retrieval.py"
    assert env.results[0].extra["line"] == 42
    assert "fuse_rrf" in env.results[0].snippet

    # HTTP 200 + isError=True + Too Many Requests -> RATE_LIMITED（不得伪装为 0 命中）
    sse_rl = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Too Many Requests (429)"}], "isError": True}}) + "\n\n"
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=sse_rl))) as cli:
        env_rl = await eng.search("fuse_rrf", limit=3, client=cli)
    assert env_rl.failure is not None
    assert env_rl.failure.reason_code == ReasonCode.RATE_LIMITED


@pytest.mark.asyncio
async def test_sourcegraph_engine_parses_sse_matches_and_rate_limit():
    """SourcegraphEngine 应解析 SSE matches 事件（0-based 行号转 1-based）及 429 限流。"""
    from omni_search.code_evidence import SourcegraphEngine

    matches = [
        {
            "type": "content",
            "repository": "github.com/org/repo",
            "path": "src/pool.py",
            "lineMatches": [{"lineNumber": 84, "line": "def fuse_results_rrf(envelopes):"}],
        }
    ]
    sse = f"event: filters\ndata: []\n\nevent: matches\ndata: {json.dumps(matches)}\n\nevent: done\ndata: {{}}\n\n"
    eng = SourcegraphEngine(env={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=sse))) as cli:
        env = await eng.search("fuse_results_rrf", limit=3, client=cli)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].extra["line"] == 85
    assert env.results[0].url == "https://sourcegraph.com/github.com/org/repo/-/blob/src/pool.py#L85"


@pytest.mark.asyncio
async def test_github_code_engine_requires_token_and_parses_matches():
    """GitHubCodeEngine 缺 GITHUB_TOKEN 时返回 MISSING_CREDENTIAL，有 token 时解析代码片段。"""
    from omni_search.code_evidence import GitHubCodeEngine

    eng_no_tok = GitHubCodeEngine(env={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500))) as cli:
        env_miss = await eng_no_tok.search("fuse_rrf", limit=3, client=cli)
    assert env_miss.failure is not None
    assert env_miss.failure.reason_code == ReasonCode.MISSING_CREDENTIAL

    payload = {
        "items": [
            {
                "name": "pool.py",
                "path": "src/pool.py",
                "html_url": "https://github.com/o/r/blob/main/src/pool.py",
                "repository": {"full_name": "o/r", "description": "demo"},
                "text_matches": [{"fragment": "def fuse_rrf(envelopes): return envelopes"}],
            }
        ]
    }
    eng_ok = GitHubCodeEngine(env={"GITHUB_TOKEN": "ghp_dummy"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))) as cli:
        env_ok = await eng_ok.search("fuse_rrf", limit=3, client=cli)
    assert env_ok.failure is None
    assert len(env_ok.results) == 1
    assert env_ok.results[0].extra["file"] == "src/pool.py"
    assert "fuse_rrf" in env_ok.results[0].snippet


@pytest.mark.asyncio
async def test_local_code_engine_runs_rg_with_lang_filter_and_ast_missing(tmp_path):
    """LocalCodeEngine 免建库实时调用本机 rg --json 并支持 lang: 过滤；缺 ast-grep 时诚实报 TOOL_MISSING。"""
    from omni_search.local_code import LocalCodeEngine

    (tmp_path / "alpha.py").write_text("def unique_alpha_symbol_99():\n    return 42\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Mention unique_alpha_symbol_99 in docs.\n", encoding="utf-8")

    eng = LocalCodeEngine(roots=[tmp_path])
    env = await eng.search("lang:py unique_alpha_symbol_99", limit=5)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].title.endswith("alpha.py")
    assert env.results[0].extra["line"] == 1
    assert "unique_alpha_symbol_99" in env.results[0].snippet

    # 若指定不存在的 ast-grep 可执行路径，ast: 查询必须诚实报 TOOL_MISSING
    eng_no_sg = LocalCodeEngine(roots=[tmp_path], sg_bin="__nonexistent_sg_binary__")
    env_ast = await eng_no_sg.search("ast:def $F(): return $X", limit=5)
    assert env_ast.failure is not None
    assert env_ast.failure.reason_code == ReasonCode.TOOL_MISSING


def test_default_engines_are_native_without_mcps_bridge_and_router_supports_code_evidence():
    """默认构建 100% 原生引擎（不含 mcps 子进程桥接），显式开启 OMNI_ENABLE_MCPS_BRIDGE=1 才挂载。"""
    from omni_search.mcp_server import build_default_engines

    native_engines = build_default_engines(env={})
    native_ids = {e.id for e in native_engines}
    assert "mcps_repo" not in native_ids
    assert "mcps_packages" not in native_ids
    assert {"github_direct", "gitlab", "codeberg", "gitee", "grep_app", "sourcegraph", "github_code", "local_code", "maven", "nuget", "rubygems", "packagist", "dockerhub", "mcp_registry"}.issubset(native_ids)

    bridged_engines = build_default_engines(env={"OMNI_ENABLE_MCPS_BRIDGE": "1"})
    bridged_ids = {e.id for e in bridged_engines}
    assert {"mcps_repo", "mcps_packages"}.issubset(bridged_ids)

    # 路由识别代码证据意图
    assert "code_evidence" in route_domains("查找函数调用代码实现示例 usage example", ["auto"])
    assert route_domains("fuse_rrf", ["code_evidence"]) == ["code_evidence"]
    assert route_domains("def main", ["local_code"]) == ["local_code"]


@pytest.mark.asyncio
async def test_all_10_expanded_yaml_sources_load_parse_and_security_route():
    """验证新增 10 个官方信源 YAML 正确加载解析，且 security 域自动路由命中 CVE/漏洞查询。"""
    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = {s.id: s for s in load_declarative_engines(engines_dir)}
    expected = {
        "github_issues",
        "ietf_rfc",
        "nvd_cve",
        "hf_datasets",
        "wikimedia_commons",
        "zenodo",
        "europe_pmc",
        "anaconda",
        "terraform_registry",
        "google_news_rss",
    }
    assert expected.issubset(specs.keys())
    assert "security" in route_domains("CVE-2026-1234 sqlite 漏洞", ["auto"])
    assert route_domains("anything", ["security"]) == ["security"]

    # 1. NVD CVE 2.0
    nvd_eng = DeclarativeEngine(specs["nvd_cve"], env={})
    nvd_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2026-9999",
                            "published": "2026-05-01T00:00:00Z",
                            "descriptions": [{"lang": "en", "value": "Heap overflow in demo parser."}],
                        }
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=nvd_mock) as cli:
        r_nvd = await nvd_eng.search("sqlite", limit=2, client=cli)
    assert r_nvd.failure is None
    assert r_nvd.results[0].url == "https://nvd.nist.gov/vuln/detail/CVE-2026-9999"
    assert "Heap overflow" in r_nvd.results[0].snippet

    # 2. IETF RFC
    rfc_eng = DeclarativeEngine(specs["ietf_rfc"], env={})
    rfc_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"objects": [{"id": 139487, "name": "rfc9110", "title": "HTTP Semantics", "abstract": "Defines HTTP semantics."}]},
        )
    )
    async with httpx.AsyncClient(transport=rfc_mock) as cli:
        r_rfc = await rfc_eng.search("http", limit=2, client=cli)
    assert r_rfc.failure is None
    assert r_rfc.results[0].url == "https://datatracker.ietf.org/doc/rfc9110/"

    # 3. Europe PMC
    epmc_eng = DeclarativeEngine(specs["europe_pmc"], env={})
    epmc_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"resultList": {"result": [{"id": "42888301", "source": "MED", "title": "CRISPR Study", "journalTitle": "Nature", "pubYear": "2026", "citedByCount": 12}]}},
        )
    )
    async with httpx.AsyncClient(transport=epmc_mock) as cli:
        r_epmc = await epmc_eng.search("crispr", limit=2, client=cli)
    assert r_epmc.failure is None
    assert r_epmc.results[0].url == "https://europepmc.org/article/MED/42888301"

    # 4. Google News RSS (XML <item>)
    gn_eng = DeclarativeEngine(specs["google_news_rss"], env={})
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><item>
      <title>SQLite 4.0 Rumors Debunked</title>
      <link>https://news.example.com/sqlite</link>
      <pubDate>Sun, 22 Nov 2026 10:00:00 GMT</pubDate>
      <description>&lt;a href="x"&gt;SQLite news summary&lt;/a&gt;</description>
    </item></channel></rss>"""
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=rss_xml))) as cli:
        r_gn = await gn_eng.search("sqlite", limit=2, client=cli)
    assert r_gn.failure is None
    assert r_gn.results[0].title == "SQLite 4.0 Rumors Debunked"
    assert r_gn.results[0].url == "https://news.example.com/sqlite"
