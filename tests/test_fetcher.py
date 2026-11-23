import httpx
import pytest

from omni_search.fetcher import safe_fetch_url


@pytest.mark.asyncio
async def test_safe_fetch_blocks_loopback_and_private_ips():
    for bad in [
        "http://127.0.0.1:8080/admin",
        "http://localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "file:///etc/passwd",
    ]:
        res = await safe_fetch_url(bad)
        assert res.ok is False
        assert res.reason_code == "ssrf-blocked"


@pytest.mark.asyncio
async def test_safe_fetch_extracts_clean_text_and_marks_untrusted():
    html = """<!doctype html>
    <html>
      <head><title>Official Spec v2</title><style>body{color:red}</style></head>
      <body>
        <nav>Home | Login</nav>
        <main>
          <h1>Rate Limits</h1>
          <p>Authenticated requests allow 30 req/min.\u200b</p>
        </main>
        <script>alert('xss')</script>
      </body>
    </html>"""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/spec", max_chars=200, client=client, resolve_host=False)

    assert res.ok is True
    assert res.untrusted_content is True
    assert res.title == "Official Spec v2"
    assert "30 req/min." in res.content
    assert "alert('xss')" not in res.content
    assert "Home | Login" not in res.content
    assert "\n\n\n" not in res.content
    assert "\u200b" not in res.content


@pytest.mark.asyncio
async def test_safe_fetch_prefers_main_article_and_collapses_whitespace_lines():
    html = """<html><head><title>Wiki</title></head><body>
    <div class="sidebar">   \n  \n  26 languages  \n  Français \n </div>
    <main>
      <div> </div><div> </div>
      <h1>Metasearch engine</h1>
      <p>A metasearch engine aggregates results from multiple search engines.</p>
    </main>
    </body></html>"""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/wiki", max_chars=400, client=client, resolve_host=False)

    assert res.ok is True
    assert "26 languages" not in res.content
    assert "Metasearch engine" in res.content
    assert "aggregates results" in res.content
    assert "\n \n" not in res.content


@pytest.mark.asyncio
async def test_safe_fetch_rejects_empty_202_challenge_body():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(202, headers={"content-type": "text/html"}, text="")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/challenge", client=client, resolve_host=False)
    assert res.ok is False
    assert res.reason_code == "empty-body"


def _build_pdf_with_title(text: str, title: str = "PDF Spec 2026") -> bytes:
    safe_t = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    safe_title = title.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe_t}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n",
        f"6 0 obj << /Title ({safe_title}) >> endobj\n".encode("latin-1"),
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xp = len(buf)
    buf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xp}\n%%EOF\n".encode("latin-1"))
    return bytes(buf)


@pytest.mark.asyncio
async def test_safe_fetch_extracts_public_pdf_text_and_title():
    pdf_bytes = _build_pdf_with_title("SQLite is Enough Hybrid Search Paper", title="SQLite Hybrid 2026")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf_bytes)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://arxiv.org/pdf/2608.24060v1", max_chars=500, client=client, resolve_host=False)

    assert res.ok is True
    assert res.untrusted_content is True
    assert res.title == "SQLite Hybrid 2026"
    assert "SQLite is Enough Hybrid Search Paper" in res.content
    assert "%PDF-1.4" not in res.content


@pytest.mark.asyncio
async def test_safe_fetch_corrupted_pdf_returns_parse_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 corrupted binary garbage")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/broken.pdf", client=client, resolve_host=False)

    assert res.ok is False
    assert res.reason_code == "pdf-parse-error"


@pytest.mark.asyncio
async def test_safe_fetch_blocks_ssrf_redirect_before_second_request():
    """302 跳转到内网 IP 时，必须在发起第二跳请求之前即阻断（仅发出 1 次请求）。"""
    seen_urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://evil.example.com/jump", client=client, resolve_host=False)

    assert res.ok is False
    assert res.reason_code == "ssrf-blocked"
    assert len(seen_urls) == 1
    assert seen_urls[0] == "https://evil.example.com/jump"


@pytest.mark.asyncio
async def test_safe_fetch_collapses_multiline_title_whitespace():
    html = "<html><head><title>智能评测\n\t\t\t| 期刊名</title></head><body><main><p>正文内容足够长。</p></main></body></html>"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/cn", client=client, resolve_host=False)

    assert res.ok is True
    assert res.title == "智能评测 | 期刊名"
    assert "\n" not in res.title and "\t" not in res.title


@pytest.mark.asyncio
async def test_safe_fetch_renders_structured_markdown_headings_code_and_tables():
    html = """<html><head><title>API Guide</title></head><body><main>
      <h1>Hybrid Search</h1>
      <p>Combine BM25 with vector search.</p>
      <h2>Example</h2>
      <ul><li>Fast FTS5</li><li>Compact vectors</li></ul>
      <pre><code>omni search "sqlite" -d local</code></pre>
      <table><tr><th>Engine</th><th>Tier</th></tr><tr><td>SearXNG</td><td>L1</td></tr></table>
    </main></body></html>"""

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=html))
    ) as client:
        res = await safe_fetch_url("https://docs.example.com/guide", max_chars=2000, client=client, resolve_host=False)

    assert res.ok is True
    assert "# Hybrid Search" in res.content
    assert "## Example" in res.content
    assert "- Fast FTS5" in res.content
    assert "```" in res.content and 'omni search "sqlite" -d local' in res.content
    assert "| Engine | Tier |" in res.content
    assert "| SearXNG | L1 |" in res.content


@pytest.mark.asyncio
async def test_safe_fetch_preserves_negotiated_text_markdown():
    seen_accept = ""

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal seen_accept
        seen_accept = req.headers.get("accept", "")
        return httpx.Response(
            200,
            headers={"content-type": "text/markdown; charset=utf-8"},
            text="# Native Markdown\n\nServed directly for AI agents.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url("https://docs.example.com/llms.txt", client=client, resolve_host=False)

    assert "text/markdown" in seen_accept
    assert res.ok is True
    assert res.source_via == "markdown-negotiated"
    assert "# Native Markdown" in res.content


@pytest.mark.asyncio
async def test_safe_fetch_maps_npmjs_url_to_registry_api():
    requested: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requested.append(str(req.url))
        if req.url.host == "registry.npmjs.org":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "name": "@inkeep/open-knowledge",
                    "description": "Open knowledge search SDK",
                    "dist-tags": {"latest": "1.2.3"},
                    "readme": "# @inkeep/open-knowledge\n\nHybrid search toolkit for agents.",
                },
            )
        return httpx.Response(403, text="Forbidden")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await safe_fetch_url(
            "https://www.npmjs.com/package/@inkeep/open-knowledge",
            client=client,
            resolve_host=False,
        )

    assert res.ok is True
    assert res.source_via == "npm-registry-api"
    assert any("registry.npmjs.org" in u for u in requested)
    assert "Hybrid search toolkit for agents." in res.content
    assert "@inkeep/open-knowledge" in res.title


@pytest.mark.asyncio
async def test_safe_fetch_maps_github_repo_to_readme_api_with_fallback():
    # 场景 1：GitHub README API 正常返回原始 Markdown
    def api_ok(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.github.com" and req.url.path == "/repos/k2-fsa/sherpa-onnx/readme":
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="# sherpa-onnx\n\nOffline ONNX speech recognition.")
        return httpx.Response(500, text="unexpected html call")

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_ok)) as client:
        res = await safe_fetch_url("https://github.com/k2-fsa/sherpa-onnx", client=client, resolve_host=False)

    assert res.ok is True
    assert res.source_via == "github-readme-api"
    assert "Offline ONNX speech recognition." in res.content
    assert "Repository navigation" not in res.content

    # 场景 2：GitHub API 限流 403 时平滑回退到常规 HTML 抓取
    def api_rl(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.github.com":
            return httpx.Response(403, text="rate limit exceeded")
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html><head><title>sherpa-onnx</title></head><body><main><p>Fallback HTML readme content.</p></main></body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_rl)) as client:
        res2 = await safe_fetch_url("https://github.com/k2-fsa/sherpa-onnx", client=client, resolve_host=False)

    assert res2.ok is True
    assert res2.source_via == "direct"
    assert "Fallback HTML readme content." in res2.content
