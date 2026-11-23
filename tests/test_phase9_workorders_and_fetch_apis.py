"""阶段 9A/9B 回归测试：research_workorders 解耦 + HuggingFace/Wikipedia 官方 API 直取。"""

from __future__ import annotations

import httpx
import pytest

from omni_search import research, research_workorders
from omni_search.fetcher import safe_fetch_url


# ---------- 9A: research_workorders 解耦与向后兼容 ----------


def test_research_workorders_reexports() -> None:
    assert research.export_research_workorders is research_workorders.export_research_workorders
    assert research.merge_lane_results is research_workorders.merge_lane_results
    assert research._response_fingerprint is research_workorders._response_fingerprint


# ---------- 9B: HuggingFace & Wikipedia 官方公开 API 直取 ----------


@pytest.mark.asyncio
async def test_fetch_huggingface_model_official_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://huggingface.co/api/models/minishlab/potion-multilingual-128M":
            return httpx.Response(
                200,
                json={
                    "id": "minishlab/potion-multilingual-128M",
                    "pipeline_tag": "feature-extraction",
                    "library_name": "model2vec",
                    "downloads": 185000,
                    "likes": 420,
                    "tags": ["model2vec", "multilingual", "static-embeddings", "license:mit"],
                    "cardData": {
                        "license": "mit",
                        "language": ["en", "zh"],
                    },
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404, text="not found")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://huggingface.co/minishlab/potion-multilingual-128M",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "huggingface-api"
    assert res.untrusted_content is True
    assert "minishlab/potion-multilingual-128M" in res.title
    assert "feature-extraction" in res.content
    assert "model2vec" in res.content
    assert "185,000" in res.content


@pytest.mark.asyncio
async def test_fetch_huggingface_dataset_official_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://huggingface.co/api/datasets/wikimedia/wikipedia":
            return httpx.Response(
                200,
                json={
                    "id": "wikimedia/wikipedia",
                    "description": "Cleaned Wikipedia dumps across 300+ languages.",
                    "downloads": 920000,
                    "likes": 1200,
                    "tags": ["multilingual", "encyclopedia"],
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404, text="not found")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://huggingface.co/datasets/wikimedia/wikipedia",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "huggingface-api"
    assert "wikimedia/wikipedia" in res.title
    assert "Cleaned Wikipedia dumps" in res.content


@pytest.mark.asyncio
async def test_fetch_wikipedia_rest_api_shortcut() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://en.wikipedia.org/api/rest_v1/page/summary/Metasearch_engine":
            return httpx.Response(
                200,
                json={
                    "title": "Metasearch engine",
                    "description": "Search tool that uses another search engine's data",
                    "extract": "A metasearch engine is an online information retrieval tool that uses the data of a web search engine to produce its own results.",
                    "content_urls": {
                        "desktop": {"page": "https://en.wikipedia.org/wiki/Metasearch_engine"}
                    },
                },
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404, text="not found")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://en.wikipedia.org/wiki/Metasearch_engine",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "wikipedia-rest-api"
    assert res.untrusted_content is True
    assert res.title == "Metasearch engine"
    assert "online information retrieval tool" in res.content
    assert "Search tool that uses another search engine's data" in res.content


@pytest.mark.asyncio
async def test_fetch_wikipedia_api_failure_falls_back_to_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/api/rest_v1/page/summary/" in str(request.url):
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            text="<html><head><title>Wiki Fallback</title></head><body><main><p>HTML fallback summary</p></main></body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
        res = await safe_fetch_url(
            "https://en.wikipedia.org/wiki/Fallback_Topic",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )

    assert res.ok is True
    assert res.source_via == "direct"
    assert "HTML fallback summary" in res.content
