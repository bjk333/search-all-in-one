"""Phase 3B 红灯测试：长文查询聚焦裁剪（omni_fetch query）+ 站点目录映射（omni map / omni_fetch mode=map）。"""

from __future__ import annotations

import httpx
import pytest

from omni_search.fetcher import safe_fetch_url
from omni_search.mcp_server import OmniMCPServer


@pytest.mark.asyncio
async def test_fetch_query_pruning_surfaces_deep_relevant_paragraphs():
    """长文档在 max_chars 预算下传入 query 时，应聚焦保留高相关深处段落而非机械截断开头。"""
    filler = "\n".join(
        f"<p>General introductory background paragraph number {i} discussing routine operations without target metrics.</p>"
        for i in range(1, 30)
    )
    html = f"""
    <html>
      <head><title>Long Engineering Handbook</title></head>
      <body>
        <article>
          <h1>Engineering Handbook</h1>
          <p>Welcome to the comprehensive platform guide.</p>
          {filler}
          <h2>Token Compression Benchmarks</h2>
          <p>Our BM25ContentFilter cuts token budget by 72% while preserving exact citation quotes.</p>
        </article>
      </body>
    </html>
    """

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://handbook.example.org/guide",
            max_chars=500,
            query="BM25ContentFilter token budget",
            client=cli,
            resolve_host=False,
        )
        assert res.ok is True
        assert res.pruned_by_query is True
        assert len(res.content) <= 500
        assert "BM25ContentFilter cuts token budget by 72%" in res.content


@pytest.mark.asyncio
async def test_site_map_extracts_llms_txt_sitemap_and_html_links_with_query_ranking():
    """safe_map_site 应按 llms.txt -> sitemap.xml -> HTML 链接发现站点目录、按 query 排序并强制 SSRF 防护。"""
    from omni_search.site_map import safe_map_site

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if u == "https://docs.example.org/llms.txt":
            return httpx.Response(
                200,
                text="# Docs\n- [Quickstart](https://docs.example.org/docs/quickstart)\n- [OAuth Guide](https://docs.example.org/docs/oauth-authentication)\n",
                headers={"content-type": "text/plain"},
            )
        if u == "https://docs.example.org/sitemap.xml":
            xml = """<?xml version="1.0" encoding="UTF-8"?>
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://docs.example.org/docs/quickstart</loc></url>
              <url><loc>https://docs.example.org/docs/pricing</loc></url>
              <url><loc>https://docs.example.org/assets/logo.png</loc></url>
            </urlset>"""
            return httpx.Response(200, text=xml, headers={"content-type": "application/xml"})
        if u == "https://fallback.example.org/":
            html = """<html><body>
              <a href="/guide/install">Install Guide</a>
              <a href="/guide/api">API Reference</a>
              <a href="/static/app.css">CSS</a>
              <a href="https://external.other.com/out">External</a>
            </body></html>"""
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        # 1) SSRF 阻断
        blocked = await safe_map_site("http://127.0.0.1/internal", client=cli, resolve_host=False)
        assert blocked.ok is False
        assert blocked.reason_code == "ssrf-blocked"

        # 2) llms.txt + sitemap 融合 + 过滤静态资源 + query 排序
        res = await safe_map_site(
            "https://docs.example.org/",
            query="oauth authentication",
            max_urls=20,
            client=cli,
            resolve_host=False,
        )
        assert res.ok is True
        assert res.untrusted_content is True
        assert "llms-txt" in res.sources_used
        assert "sitemap" in res.sources_used
        urls = [e.url for e in res.urls]
        assert not any(u.endswith(".png") for u in urls)
        assert urls[0] == "https://docs.example.org/docs/oauth-authentication"

        # 3) llms.txt & sitemap 均 404 时平滑回退同源 HTML 链接
        fb = await safe_map_site("https://fallback.example.org/", client=cli, resolve_host=False)
        assert fb.ok is True
        assert "html-links" in fb.sources_used
        fb_urls = {e.url for e in fb.urls}
        assert "https://fallback.example.org/guide/install" in fb_urls
        assert "https://fallback.example.org/guide/api" in fb_urls
        assert not any("external.other.com" in u or u.endswith(".css") for u in fb_urls)


@pytest.mark.asyncio
async def test_mcp_omni_fetch_map_mode_keeps_6_orthogonal_tools(tmp_path):
    """MCP 保持 6 个正交工具，同时 omni_fetch 暴露 query 与 mode(fetch/map) 参数。"""
    srv = OmniMCPServer(health_path=tmp_path / "h.json")
    try:
        tl = await srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = {t["name"]: t for t in tl["result"]["tools"]}
        assert len(tools) >= 6
        assert "omni_map" not in tools  # map 聚合在 omni_fetch(mode='map') 内保持工具面正交
        props = tools["omni_fetch"]["inputSchema"]["properties"]
        assert "query" in props
        assert "mode" in props
        assert "maxUrls" in props
    finally:
        await srv.aclose()
