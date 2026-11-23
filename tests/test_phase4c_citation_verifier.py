"""阶段 4C 红灯测试：主张级引用核验引擎 (verifier.py) + MCP omni_verify。"""

from __future__ import annotations

import pytest

from omni_search.fetcher import FetchResult
from omni_search.mcp_server import OmniMCPServer
from omni_search.verifier import (
    ClaimInput,
    extract_claims_from_markdown,
    render_verification_markdown,
    verify_claims,
)


def test_extract_claims_from_markdown_skips_code_fences_and_extracts_links() -> None:
    md = """
# 调研结论

- SQLite FTS5 支持 BM25 排序与前缀查询，见 [SQLite 官方文档](https://www.sqlite.org/fts5.html)。
- 某模型在 2026 年达到 99.9% 准确率（参考 https://example.com/paper）。
```python
# 代码块中的链接不应被误抽为引用主张
url = "https://example.com/ignored-in-code"
```
"""
    claims = extract_claims_from_markdown(md)
    urls = [c.url for c in claims]
    assert "https://www.sqlite.org/fts5.html" in urls
    assert "https://example.com/paper" in urls
    assert "https://example.com/ignored-in-code" not in urls
    fts_claim = next(c for c in claims if "sqlite.org" in c.url)
    assert "FTS5" in fts_claim.claim and "BM25" in fts_claim.claim


@pytest.mark.asyncio
async def test_verify_claims_distinguishes_supported_partial_unsupported_unreachable_blocked() -> None:
    pages = {
        "https://docs.example.com/fts5": FetchResult(
            ok=True,
            url="https://docs.example.com/fts5",
            status_code=200,
            title="SQLite FTS5 Extension",
            content="SQLite FTS5 is a virtual table module that provides full-text search with built-in BM25 ranking and prefix queries.",
            untrusted_content=True,
        ),
        "https://docs.example.com/release": FetchResult(
            ok=True,
            url="https://docs.example.com/release",
            status_code=200,
            title="Release Notes",
            content="Version v2.0.0 introduces async streaming support and HTTP/2 pooling.",
            untrusted_content=True,
        ),
        "https://docs.example.com/cooking": FetchResult(
            ok=True,
            url="https://docs.example.com/cooking",
            status_code=200,
            title="Pasta Recipe",
            content="Boil water with salt for 10 minutes and add fresh basil leaves with olive oil.",
            untrusted_content=True,
        ),
        "https://docs.example.com/dead": FetchResult(
            ok=False,
            url="https://docs.example.com/dead",
            status_code=404,
            reason_code="http-error",
            reason="HTTP 404",
            untrusted_content=True,
        ),
        "http://127.0.0.1:9000/internal": FetchResult(
            ok=False,
            url="http://127.0.0.1:9000/internal",
            reason_code="ssrf-blocked",
            reason="private ip blocked",
            untrusted_content=True,
        ),
    }

    async def _fake_fetcher(url: str, max_chars: int = 12000, timeout_sec: float = 15.0, query: str = "") -> FetchResult:
        return pages[url]

    report = await verify_claims(
        claims=[
            ClaimInput(claim="SQLite FTS5 provides built-in BM25 ranking and full-text search", url="https://docs.example.com/fts5"),
            ClaimInput(claim="Version v9.9.9 introduces async streaming support and quantum encryption", url="https://docs.example.com/release"),
            ClaimInput(claim="CVE-2026-9999 allows remote code execution in OpenSSL 3.5", url="https://docs.example.com/cooking"),
            ClaimInput(claim="Dead citation link claim", url="https://docs.example.com/dead"),
            ClaimInput(claim="Internal metadata endpoint claim", url="http://127.0.0.1:9000/internal"),
        ],
        fetcher=_fake_fetcher,
    )

    assert report.total_claims == 5
    assert report.untrusted_content is True
    by_url = {it.url: it for it in report.items}
    assert by_url["https://docs.example.com/fts5"].verdict == "supported"
    assert len(by_url["https://docs.example.com/fts5"].supporting_quotes) >= 1
    assert by_url["https://docs.example.com/release"].verdict == "partial"
    assert any("v9.9.9" in t.lower() or "9.9.9" in t for t in by_url["https://docs.example.com/release"].missing_terms)
    assert by_url["https://docs.example.com/cooking"].verdict == "unsupported"
    assert by_url["https://docs.example.com/dead"].verdict == "unreachable"
    assert by_url["http://127.0.0.1:9000/internal"].verdict == "blocked"
    assert report.supported_count == 1
    assert report.partial_count == 1
    assert report.unsupported_count == 1
    assert report.unreachable_count == 1
    assert report.blocked_count == 1

    md = render_verification_markdown(report)
    assert "引用核验审计报告" in md
    assert "unsupported" in md
    assert "supported" in md


@pytest.mark.asyncio
async def test_mcp_omni_verify_exposes_tool_and_blocks_ssrf() -> None:
    srv = OmniMCPServer(env={})
    try:
        listed = await srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in listed["result"]["tools"]}
        assert "omni_verify" in names

        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "omni_verify",
                    "arguments": {
                        "markdown": "- 本地内网服务暴露密钥 [internal](http://127.0.0.1:8080/secret)",
                        "renderMarkdown": True,
                    },
                },
            }
        )
        sc = resp["result"]["structuredContent"]
        assert sc["total_claims"] == 1
        assert sc["blocked_count"] == 1
        assert sc["untrusted_content"] is True
        assert "引用核验审计报告" in sc["markdown"]
    finally:
        await srv.aclose()
