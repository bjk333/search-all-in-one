"""阶段 8A/8B/8C 回归测试：html_markdown 解耦、arXiv/crates.io 官方 API 直取、doctor 精准按引擎重置。"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omni_search import fetcher, html_markdown
from omni_search.cli import main as cli_main
from omni_search.contracts import ReasonCode
from omni_search.fetcher import safe_fetch_url
from omni_search.health import HealthStore
from omni_search.mcp_server import OmniMCPServer, _TOOLS_SPEC


# ---------- 8A: html_markdown 解耦与向后兼容 ----------


def test_html_markdown_module_reexports() -> None:
    assert fetcher._CleanTextExtractor is html_markdown._CleanTextExtractor
    assert fetcher.prune_content_for_query is html_markdown.prune_content_for_query
    assert fetcher._normalize_extracted_text is html_markdown._normalize_extracted_text


# ---------- 8B: arXiv & crates.io 官方公开 API 直取 ----------


@pytest.mark.asyncio
async def test_fetch_arxiv_official_api_shortcut() -> None:
    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2508.14704v2</id>
    <published>2025-08-20T12:00:00Z</published>
    <updated>2025-08-22T09:00:00Z</updated>
    <title>Hybrid Retrieval with Local SQLite FTS5 and Static Vectors</title>
    <summary>We present an auditable hybrid search architecture combining BM25F and reciprocal rank fusion.</summary>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Lin</name></author>
    <category term="cs.IR" />
    <category term="cs.CL" />
    <link title="pdf" href="https://arxiv.org/pdf/2508.14704v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org/api/query" in str(request.url) and "2508.14704v2" in str(request.url):
            return httpx.Response(200, text=atom_xml, headers={"content-type": "application/atom+xml"})
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        res = await safe_fetch_url(
            "https://arxiv.org/abs/2508.14704v2",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "arxiv-api"
    assert res.untrusted_content is True
    assert "Hybrid Retrieval" in res.title
    assert "Alice Zhang" in res.content and "Bob Lin" in res.content
    assert "2025-08-20" in res.content
    assert "cs.IR" in res.content
    assert "reciprocal rank fusion" in res.content
    assert "https://arxiv.org/pdf/2508.14704v2" in res.content


@pytest.mark.asyncio
async def test_fetch_crates_io_official_api_shortcut() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://crates.io/api/v1/crates/serde":
            return httpx.Response(
                200,
                json={
                    "crate": {
                        "id": "serde",
                        "name": "serde",
                        "max_stable_version": "1.0.219",
                        "description": "A generic serialization/deserialization framework",
                        "repository": "https://github.com/serde-rs/serde",
                        "documentation": "https://docs.rs/serde",
                        "downloads": 350000000,
                    }
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        res = await safe_fetch_url(
            "https://crates.io/crates/serde",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "crates-io-api"
    assert res.final_url == "https://crates.io/api/v1/crates/serde"
    assert res.untrusted_content is True
    assert "serde" in res.title
    assert "1.0.219" in res.content
    assert "serialization/deserialization" in res.content
    assert "https://github.com/serde-rs/serde" in res.content


@pytest.mark.asyncio
async def test_fetch_arxiv_api_failure_falls_back_to_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(
            200,
            text="<html><head><title>Fallback arXiv Page</title></head><body><main><p>Fallback abstract text</p></main></body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        res = await safe_fetch_url(
            "https://arxiv.org/abs/2508.99999",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "direct"
    assert "Fallback abstract text" in res.content


# ---------- 8C: doctor 精准按引擎重置健康状态 ----------


@pytest.mark.asyncio
async def test_doctor_targeted_reset_engines_preserves_other_cooldowns(tmp_path: pathlib.Path) -> None:
    hpath = tmp_path / "health.json"
    hs = HealthStore(hpath)
    hs.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED)
    for _ in range(4):
        hs.record_failure("codeberg", ReasonCode.DEADLINE)
        hs.record_failure("gitee", ReasonCode.DEADLINE)

    server = OmniMCPServer(health_path=hpath, env={})
    try:
        # 仅精准重置 codeberg 与 gitee，不碰 deepseek
        res = await server.doctor(reset_engines=["codeberg", "gitee"])
        assert res["reset_applied"] == ["codeberg", "gitee"]
        assert server.health_store.check("codeberg").consecutive_failures == 0
        assert server.health_store.check("gitee").consecutive_failures == 0
        # deepseek 仍在配额耗尽冷却中
        ds = server.health_store.check("deepseek")
        assert ds.in_cooldown is True
        assert ds.last_reason == ReasonCode.QUOTA_EXHAUSTED.value

        # 全量重置应返回 "all"
        res_all = await server.doctor(reset_health=True)
        assert res_all["reset_applied"] == "all"
        assert server.health_store.check("deepseek").in_cooldown is False
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_doctor_recommendations_suggest_targeted_reset_engines(tmp_path: pathlib.Path) -> None:
    hpath = tmp_path / "health.json"
    hs = HealthStore(hpath)
    hs.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED)
    for _ in range(4):
        hs.record_failure("codeberg", ReasonCode.DEADLINE)

    server = OmniMCPServer(health_path=hpath, env={})
    try:
        res = await server.doctor()
        joined = "\n".join(res["recommendations"])
        assert "--reset-engines codeberg" in joined
        assert "--reset-engines deepseek" in joined
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_mcp_doctor_schema_and_call_with_reset_engines(tmp_path: pathlib.Path) -> None:
    doc_tool = next(t for t in _TOOLS_SPEC if t["name"] == "omni_doctor")
    assert "resetEngines" in doc_tool["inputSchema"]["properties"]

    hpath = tmp_path / "health.json"
    hs = HealthStore(hpath)
    hs.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED)
    for _ in range(4):
        hs.record_failure("gitee", ReasonCode.DEADLINE)

    server = OmniMCPServer(health_path=hpath, env={})
    try:
        resp = await server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 81,
                "method": "tools/call",
                "params": {"name": "omni_doctor", "arguments": {"resetEngines": ["gitee"]}},
            }
        )
        payload = resp["result"]["structuredContent"]
        assert payload["reset_applied"] == ["gitee"]
        assert server.health_store.check("gitee").consecutive_failures == 0
        assert server.health_store.check("deepseek").in_cooldown is True
    finally:
        await server.aclose()


def test_cli_doctor_reset_engines_flag(capsys: Any) -> None:
    fake_doctor = AsyncMock(
        return_value={
            "version": "0.8.0",
            "engine_count": 0,
            "domains": [],
            "disabled_engines": [],
            "reset_applied": ["codeberg", "gitee"],
            "engines": [],
        }
    )
    with patch("omni_search.mcp_server.OmniMCPServer.doctor", fake_doctor):
        with pytest.raises(SystemExit) as exc:
            cli_main(["doctor", "--reset-engines", "codeberg,gitee", "--compact"])
        assert exc.value.code == 0

    fake_doctor.assert_awaited_once()
    kwargs = fake_doctor.await_args.kwargs
    assert kwargs["reset_engines"] == ["codeberg", "gitee"]
    out = json.loads(capsys.readouterr().out)
    assert out["reset_applied"] == ["codeberg", "gitee"]
