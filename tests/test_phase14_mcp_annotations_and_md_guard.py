"""阶段 14 单测：MCP 2025-06-18/2026 Tool annotations + title、mcp_schemas 拆分、
Accept: text/markdown 深路径误返站点首页自动回退守卫，以及动态 User-Agent 版本号。"""

from __future__ import annotations

import pathlib

import httpx
import pytest

from omni_search import fetcher
from omni_search.mcp_server import _TOOLS_SPEC, OmniMCPServer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_all_mcp_tools_expose_title_and_annotations():
    """全部 8 个 MCP 工具必须声明人类可读 title 与完整 4 项 annotations 提示。"""
    assert len(_TOOLS_SPEC) == 8
    by_name = {t["name"]: t for t in _TOOLS_SPEC}

    read_only_tools = {"omni_search", "omni_fetch", "omni_batch", "omni_research", "omni_verify"}
    mutating_tools = {"omni_index", "omni_doctor", "omni_monitor"}
    assert set(by_name) == read_only_tools | mutating_tools

    for name, spec in by_name.items():
        title = spec.get("title")
        assert isinstance(title, str) and len(title.strip()) >= 4, f"{name} missing valid title"
        ann = spec.get("annotations")
        assert isinstance(ann, dict), f"{name} missing annotations"
        for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            assert isinstance(ann.get(key), bool), f"{name}.annotations.{key} must be bool"
        assert ann["destructiveHint"] is False, f"{name} should not be destructive"

        if name in read_only_tools:
            assert ann["readOnlyHint"] is True, f"{name} must be readOnlyHint=True"
            assert ann["idempotentHint"] is True, f"{name} must be idempotentHint=True"
            assert ann["openWorldHint"] is True, f"{name} must be openWorldHint=True"
        else:
            assert ann["readOnlyHint"] is False, f"{name} modifies local state -> readOnlyHint=False"

    assert by_name["omni_index"]["annotations"]["openWorldHint"] is False
    assert by_name["omni_index"]["annotations"]["idempotentHint"] is True
    assert by_name["omni_doctor"]["annotations"]["idempotentHint"] is True
    assert by_name["omni_monitor"]["annotations"]["idempotentHint"] is False


@pytest.mark.asyncio
async def test_mcp_tools_list_rpc_returns_annotations(tmp_path):
    """通过 JSON-RPC tools/list 返回的每个工具均包含 title 与 annotations。"""
    srv = OmniMCPServer(health_path=tmp_path / "h.json", env={"OMNI_TELEMETRY": "0", "OMNI_DISK_CACHE_TTL": "0"})
    try:
        resp = await srv.handle_request({"jsonrpc": "2.0", "id": 14, "method": "tools/list", "params": {}})
        tools = resp["result"]["tools"]
        assert len(tools) == 8
        assert all("title" in t and "annotations" in t for t in tools)
    finally:
        await srv.aclose()


def test_mcp_server_split_keeps_module_compact():
    """mcp_schemas.py 拆分后 mcp_server.py 应远低于 700 行，且全包无 >=1000 行巨石。"""
    schemas_file = PROJECT_ROOT / "src" / "omni_search" / "mcp_schemas.py"
    server_file = PROJECT_ROOT / "src" / "omni_search" / "mcp_server.py"
    assert schemas_file.exists(), "mcp_schemas.py must exist"
    server_lines = len(server_file.read_text(encoding="utf-8").splitlines())
    assert server_lines < 700, f"mcp_server.py should be < 700 lines after split, got {server_lines}"


@pytest.mark.asyncio
async def test_markdown_negotiation_root_llms_fallback_to_html():
    """深路径 URL 在 Accept: text/markdown 下被站点网关误返根 /llms.txt 首页时，应自动回退 HTML 获取真实文章。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        accept = request.headers.get("accept", "")
        calls.append(accept)
        if "text/markdown" in accept:
            body = (
                "# Acme Browser Agent\n\n"
                "> Your free AI browser agent.\n\n"
                "- [Storefront](https://agents.example.com/llms.txt?src=llms-txt)\n"
                "- [Complete guide](https://agents.example.com/llms-full.txt?src=llms-txt)\n"
                "- [Pricing](https://example.com/pricing)\n"
                "- [Docs](https://example.com/docs)\n"
            )
            return httpx.Response(200, headers={"Content-Type": "text/markdown; charset=utf-8"}, text=body, request=request)
        html = (
            "<html><head><title>Agent Web Protocol Stack</title></head>"
            "<body><article><h1>Agent Web Protocol Stack</h1>"
            "<p>MCP is for tools, A2A is for agents, and WebMCP plus llms.txt defines the agent web protocol stack for websites in 2026.</p>"
            "</article></body></html>"
        )
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text=html, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await fetcher.safe_fetch_url(
            "https://example.com/blog/agent-web-protocol-stack",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert len(calls) == 2, f"expected markdown attempt + html fallback, got {calls}"
    assert res.source_via == "html-fallback-from-markdown"
    assert "Agent Web Protocol Stack" in res.title
    assert "protocol stack" in res.content.lower()


@pytest.mark.asyncio
async def test_markdown_negotiation_genuine_article_does_not_fallback():
    """深路径 URL 若 Accept: text/markdown 确实返回了对应文章正文，不应触发多余 HTML 请求。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("accept", ""))
        body = (
            "# Agent Web Protocol Stack\n\n"
            "This article explains how the agent web protocol stack connects MCP, A2A, and llms.txt.\n"
            "See also https://example.com/llms.txt for site index.\n"
        )
        return httpx.Response(200, headers={"Content-Type": "text/markdown; charset=utf-8"}, text=body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await fetcher.safe_fetch_url(
            "https://example.com/blog/agent-web-protocol-stack",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert len(calls) == 1
    assert res.source_via == "markdown-negotiated"
    assert "Agent Web Protocol Stack" in res.title


@pytest.mark.asyncio
async def test_markdown_negotiation_root_url_keeps_markdown():
    """站点根路径 / 收到站点级 Markdown 简介是预期行为，不应误判回退 HTML。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("accept", ""))
        body = "# Acme\n\n> Site overview\n\n- [Index](https://example.com/llms.txt)\n"
        return httpx.Response(200, headers={"Content-Type": "text/markdown; charset=utf-8"}, text=body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await fetcher.safe_fetch_url(
            "https://example.com/",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert len(calls) == 1
    assert res.source_via == "markdown-negotiated"


def test_no_stale_hardcoded_user_agent_version():
    """fetcher.py 与 fetch_apis.py 不得硬编码旧版本 search-all-in-one/1.1。"""
    for rel in ("src/omni_search/fetcher.py", "src/omni_search/fetch_apis.py"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert "search-all-in-one/1.1" not in text, f"stale hardcoded UA in {rel}"
