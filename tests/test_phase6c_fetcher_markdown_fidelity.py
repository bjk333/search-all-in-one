"""阶段 6C 红灯测试：fetcher 代码块语言围栏、meta title/description 兜底与外链保真。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.fetcher import safe_fetch_url


@pytest.mark.asyncio
async def test_fetcher_preserves_code_language_external_links_and_meta_fallback() -> None:
    html_doc = """
    <html>
      <head>
        <meta property="og:title" content="MCP Streamable HTTP Spec" />
        <meta name="description" content="Official transport specification for Streamable HTTP." />
      </head>
      <body>
        <article>
          <h1>Overview</h1>
          <p>Read the <a href="https://spec.modelcontextprotocol.io/http">official spec</a> for details.</p>
          <pre><code class="language-python">async def connect():\n    return True</code></pre>
        </article>
      </body>
    </html>
    """
    spa_doc = """
    <html>
      <head>
        <title>SPA App</title>
        <meta name="description" content="Fallback meta summary when body is empty JS shell." />
      </head>
      <body><div id="root"></div><script>window.__APP__=1;</script></body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/spec":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html_doc)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=spa_doc)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as cli:
        r1 = await safe_fetch_url("https://docs.example.com/spec", client=cli, resolve_host=False)
        r2 = await safe_fetch_url("https://docs.example.com/spa", client=cli, resolve_host=False)

    assert r1.ok is True
    assert r1.title == "MCP Streamable HTTP Spec"
    assert "```python" in r1.content
    assert "[official spec](https://spec.modelcontextprotocol.io/http)" in r1.content

    assert r2.ok is True
    assert "Fallback meta summary" in r2.content
