"""阶段 11A–11C 回归测试：local_query & fetch_apis 解耦、全模块千行门禁、DOI/HN 官方 API 直取、/llms-full.txt 回退与 Markdown 协商标题抽取。"""

from __future__ import annotations

import pathlib

import httpx
import pytest

from omni_search import fetch_apis, fetcher, local_index, local_query

_SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "omni_search"


def test_phase11_module_reexports_and_sub_1000_line_guardrail() -> None:
    # 1) 向后兼容 re-export 契约
    assert local_index.tokenize_bilingual is local_query.tokenize_bilingual
    assert local_index._query_db_sync is local_query._query_db_sync
    assert local_index._resolve_related_docs is local_query._resolve_related_docs
    assert local_index._score_chunk_rows is local_query._score_chunk_rows
    assert fetcher._try_fetch_official_api is fetch_apis._try_fetch_official_api

    # 2) 全仓库所有源码模块严格低于 1,000 行
    oversized = []
    for py_file in sorted(_SRC_DIR.glob("*.py")):
        line_count = len(py_file.read_text(encoding="utf-8").splitlines())
        if line_count >= 1000:
            oversized.append(f"{py_file.name}:{line_count}")
    assert not oversized, f"modules >= 1000 lines: {oversized}"


@pytest.mark.asyncio
async def test_doi_crossref_and_arxiv_doi_official_api_fastpath() -> None:
    arxiv_atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Towards an Agent-First Web</title>
        <summary>Redesigning web protocols for autonomous AI agents.</summary>
        <published>2026-06-17T00:00:00Z</published>
        <author><name>Ada Lovelace</name></author>
        <category term="cs.AI" />
        <link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2606.19116" />
      </entry>
    </feed>"""

    crossref_json = {
        "message": {
            "title": ["Citation Graph-Based Multi-Hop Retrieval"],
            "container-title": ["Journal of Applied Informatics"],
            "issued": {"date-parts": [[2026, 5, 1]]},
            "is-referenced-by-count": 42,
            "author": [{"given": "Grace", "family": "Hopper"}],
            "URL": "https://example.org/paper/10.30871",
            "abstract": "<jats:p>Hybrid RRF improves <jats:italic>recall</jats:italic> across legal corpora.</jats:p>",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://export.arxiv.org/api/query?id_list=2606.19116"):
            return httpx.Response(200, content=arxiv_atom.encode("utf-8"), headers={"content-type": "application/atom+xml"})
        if url.startswith("https://api.crossref.org/works/10.30871/jaic.v10i5.14131"):
            return httpx.Response(200, json=crossref_json, headers={"content-type": "application/json"})
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        # arXiv DOI 优先直连 arXiv Atom API
        r_arxiv = await fetcher.safe_fetch_url(
            "https://doi.org/10.48550/arxiv.2606.19116",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r_arxiv.ok is True
        assert r_arxiv.source_via == "arxiv-api"
        assert r_arxiv.title == "Towards an Agent-First Web"
        assert "Redesigning web protocols" in r_arxiv.content
        assert r_arxiv.untrusted_content is True

        # 通用 DOI 走 Crossref Works API 并清理 JATS XML 标签
        r_cr = await fetcher.safe_fetch_url(
            "https://doi.org/10.30871/jaic.v10i5.14131",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r_cr.ok is True
        assert r_cr.source_via == "crossref-doi-api"
        assert r_cr.title == "Citation Graph-Based Multi-Hop Retrieval"
        assert "Citations: 42" in r_cr.content
        assert "Grace Hopper" in r_cr.content
        assert "<jats:" not in r_cr.content
        assert "Hybrid RRF improves recall across legal corpora." in r_cr.content


@pytest.mark.asyncio
async def test_hackernews_official_firebase_api_fastpath() -> None:
    hn_item = {
        "id": 42000001,
        "type": "story",
        "by": "pg",
        "title": "Show HN: Local-First Agentic Search",
        "score": 321,
        "descendants": 87,
        "url": "https://example.com/omni-search",
        "text": "Built with &lt;4 deps&gt; and <i>zero</i> browser overhead.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://hacker-news.firebaseio.com/v0/item/42000001.json":
            return httpx.Response(200, json=hn_item, headers={"content-type": "application/json"})
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        res = await fetcher.safe_fetch_url(
            "https://news.ycombinator.com/item?id=42000001",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert res.ok is True
        assert res.source_via == "hackernews-api"
        assert res.title == "Show HN: Local-First Agentic Search"
        assert "Score: 321" in res.content
        assert "Comments: 87" in res.content
        assert "https://example.com/omni-search" in res.content
        assert "<4 deps>" in res.content
        assert "<i>" not in res.content


@pytest.mark.asyncio
async def test_llms_full_txt_fallback_and_markdown_negotiated_title() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://docs.example.com/llms.txt":
            return httpx.Response(404, text="Not Found")
        if url == "https://docs.example.com/llms-full.txt":
            return httpx.Response(
                200,
                text="# Full Docs Index\n\nComplete API reference for agents.",
                headers={"content-type": "text/markdown"},
            )
        if url == "https://agent.example.com/guide":
            return httpx.Response(
                200,
                text="---\ntitle: \"Agentic Web Guide 2026\"\n---\n\n# Ignored H1\n\nBody paragraph here.",
                headers={"content-type": "text/markdown; charset=utf-8"},
            )
        if url == "https://agent.example.com/h1-only":
            return httpx.Response(
                200,
                text="# Native Markdown Page\n\nClean content without HTML noise.",
                headers={"content-type": "text/markdown"},
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        r_full = await fetcher.safe_fetch_url(
            "https://docs.example.com/",
            client=cli,
            resolve_host=False,
            use_cache=False,
            prefer_llms_txt=True,
        )
        assert r_full.ok is True
        assert r_full.source_via == "llms-full-txt"
        assert r_full.final_url == "https://docs.example.com/llms-full.txt"
        assert "Complete API reference" in r_full.content

        r_fm = await fetcher.safe_fetch_url(
            "https://agent.example.com/guide",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r_fm.ok is True
        assert r_fm.source_via == "markdown-negotiated"
        assert r_fm.title == "Agentic Web Guide 2026"

        r_h1 = await fetcher.safe_fetch_url(
            "https://agent.example.com/h1-only",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r_h1.ok is True
        assert r_h1.source_via == "markdown-negotiated"
        assert r_h1.title == "Native Markdown Page"
