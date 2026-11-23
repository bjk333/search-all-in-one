import pathlib
import httpx
import pytest

from omni_search.contracts import ReasonCode
from omni_search.declarative import (
    DeclarativeEngine,
    DeclarativeSpec,
    extract_path,
    load_declarative_engines,
)


def test_extract_path_supports_nested_list_and_fallback():
    obj = {
        "a": {"b": [{"name": "first"}, {"name": "second"}]},
        "doi": "",
        "id": "https://openalex.org/W1",
    }
    assert extract_path(obj, "a.b[0].name") == "first"
    assert extract_path(obj, "a.b[*].name") == ["first", "second"]
    assert extract_path(obj, "doi || id") == "https://openalex.org/W1"


def test_load_bundled_yaml_engines_validates_all():
    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = load_declarative_engines(engines_dir)
    ids = {s.id for s in specs}
    assert {
        "wikipedia",
        "wikipedia_zh",
        "openlibrary",
        "stackexchange",
        "hackernews",
        "openalex",
        "arxiv",
        "searxng",
        "brave",
        "deepseek",
        "ddg_instant",
    }.issubset(ids)
    assert len(specs) >= 18


@pytest.mark.asyncio
async def test_declarative_engine_missing_required_credential_returns_blocked():
    spec = DeclarativeSpec(
        id="brave",
        domain="web",
        tier="L2",
        protocol="http_json",
        auth={"env": "BRAVE_API_KEY", "required": True, "placement": "header:X-Subscription-Token"},
        request={"method": "GET", "url": "https://api.search.brave.com/res/v1/web/search", "params": {"q": "{query}"}},
        response={"items_path": "web.results", "output_map": {"title": "title", "url": "url"}},
    )
    engine = DeclarativeEngine(spec, env={})
    async with httpx.AsyncClient() as client:
        env = await engine.search("rust", limit=5, client=client)
    assert env.failure is not None
    assert env.failure.reason_code == ReasonCode.MISSING_CREDENTIAL
    assert env.requests.attempted == 0


@pytest.mark.asyncio
async def test_declarative_engine_http_json_success_and_rate_limit():
    spec = DeclarativeSpec(
        id="hn",
        domain="news",
        tier="L0",
        protocol="http_json",
        request={"method": "GET", "url": "https://hn.algolia.com/api/v1/search", "params": {"query": "{query}", "hitsPerPage": "{limit}"}},
        response={
            "items_path": "hits",
            "output_map": {
                "id": "objectID",
                "title": "title",
                "url": "url || story_url",
                "popularity": "points",
                "popularity_unit": "'points'",
            },
        },
    )

    def ok_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"hits": [{"objectID": "42", "title": "OmniSearch Launch", "url": "https://example.com/omni?token=secret", "points": 321}]},
        )

    engine = DeclarativeEngine(spec, env={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler)) as client:
        env = await engine.search("omni", limit=5, client=client)

    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].title == "OmniSearch Launch"
    assert "secret" not in env.results[0].url
    assert env.results[0].popularity == 321
    assert env.results[0].unit == "points"

    def rl_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "12"}, text="Too Many Requests")

    async with httpx.AsyncClient(transport=httpx.MockTransport(rl_handler)) as client:
        env_rl = await engine.search("omni", limit=5, client=client)
    assert env_rl.failure is not None
    assert env_rl.failure.reason_code == ReasonCode.RATE_LIMITED


@pytest.mark.asyncio
async def test_declarative_engine_atom_xml_parses_arxiv():
    spec = DeclarativeSpec(
        id="arxiv",
        domain="academic",
        tier="L0",
        protocol="http_atom",
        request={"method": "GET", "url": "https://export.arxiv.org/api/query", "params": {"search_query": "all:{query}", "max_results": "{limit}"}},
        response={
            "items_path": "entry",
            "output_map": {
                "id": "id",
                "title": "title",
                "url": "id",
                "snippet": "summary",
                "published_at": "published",
            },
        },
    )
    xml_body = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2601.02365v1</id>
        <title>FUSE: Failure-aware MultiModal Search</title>
        <summary>Failure-aware evidence fusion.</summary>
        <published>2026-01-05T00:00:00Z</published>
      </entry>
    </feed>"""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=xml_body)

    engine = DeclarativeEngine(spec, env={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        env = await engine.search("fuse", limit=3, client=client)

    assert env.failure is None
    assert len(env.results) == 1
    assert "FUSE" in env.results[0].title
    assert env.results[0].url == "http://arxiv.org/abs/2601.02365v1"


@pytest.mark.asyncio
async def test_declarative_engine_deepseek_402_truncation_int_limit_and_min_score():
    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = {s.id: s for s in load_declarative_engines(engines_dir)}
    ds_spec = specs["deepseek"]
    engine = DeclarativeEngine(ds_spec, env={"DEEPSEEK_API_KEY": "sk-test-dummy"})

    captured: dict = {}

    def ok_handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(req.content.decode("utf-8"))
        captured["auth"] = req.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "High Quality", "url": "https://example.com/good", "summary": "Good hit", "rerank_score": 2.4, "date": "2026-11-20"},
                    {"title": "Low Noise", "url": "https://example.com/noise", "summary": "Noise", "rerank_score": 0.3, "date": "2026-11-20"},
                ]
            },
        )

    long_query = "a" * 150
    async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler)) as client:
        env = await engine.search(long_query, limit=4, client=client)

    assert captured["auth"] == "Bearer sk-test-dummy"
    assert len(captured["body"]["query"]) == 128
    assert captured["body"]["count"] == 4
    assert isinstance(captured["body"]["count"], int)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].title == "High Quality"
    assert any("128" in w for w in env.warnings)
    assert any("filtered" in w.lower() or "丢弃" in w for w in env.warnings)

    # 全部低于门槛 -> filtered（不是 zero-hit）
    def all_noise_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"title": "Noise", "url": "https://x.test/1", "rerank_score": 0.2}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(all_noise_handler)) as client:
        env_flt = await engine.search("noise", limit=3, client=client)
    assert env_flt.failure is not None
    assert env_flt.failure.reason_code == ReasonCode.FILTERED
    assert env_flt.results == []

    # 402 余额不足 -> quota-exhausted
    def pay_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(402, text="Insufficient Balance")

    async with httpx.AsyncClient(transport=httpx.MockTransport(pay_handler)) as client:
        env_402 = await engine.search("test", limit=3, client=client)
    assert env_402.failure is not None
    assert env_402.failure.reason_code == ReasonCode.QUOTA_EXHAUSTED


@pytest.mark.asyncio
async def test_mdn_npm_and_crates_direct_yaml_specs_parse():
    """验证新增的 3 个高速 L0 官方直连 YAML（mdn / npm_direct / crates_direct）正确加载并映射。"""
    import pathlib
    from omni_search.declarative import load_declarative_engines

    engines_dir = pathlib.Path(__file__).resolve().parent.parent / "engines"
    specs = {s.id: s for s in load_declarative_engines(engines_dir)}
    assert {"mdn", "npm_direct", "crates_direct"}.issubset(specs.keys())

    # MDN url_template 拼接
    mdn_eng = DeclarativeEngine(specs["mdn"], env={})
    mdn_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "slug": "web/api/fetch_api",
                        "title": "Fetch API",
                        "mdn_url": "/en-US/docs/Web/API/Fetch_API",
                        "summary": "Interface for fetching resources.",
                        "score": 43.7,
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=mdn_mock) as client:
        r_mdn = await mdn_eng.search("fetch", limit=2, client=client)
    assert r_mdn.failure is None
    assert r_mdn.results[0].url == "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"

    # MDN 低分无关噪声（score=14.9 < 25.0）应被声明式 min_score 拦截为 FILTERED
    mdn_noise = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"documents": [{"slug": "css/font", "title": "font-synthesis", "mdn_url": "/en-US/docs/Web/CSS/font", "summary": "CSS", "score": 14.9}]},
        )
    )
    async with httpx.AsyncClient(transport=mdn_noise) as client:
        r_mdn_noise = await mdn_eng.search("大模型记忆", limit=2, client=client)
    assert r_mdn_noise.results == []
    assert r_mdn_noise.failure is not None
    assert r_mdn_noise.failure.reason_code == ReasonCode.FILTERED

    # npm_direct
    npm_eng = DeclarativeEngine(specs["npm_direct"], env={})
    npm_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "objects": [
                    {
                        "downloads": {"monthly": 1122428},
                        "package": {
                            "name": "sqlite",
                            "description": "SQLite client for Node.js",
                            "date": "2023-11-01T23:59:12Z",
                            "links": {"npm": "https://www.npmjs.com/package/sqlite"},
                        },
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=npm_mock) as client:
        r_npm = await npm_eng.search("sqlite", limit=2, client=client)
    assert r_npm.failure is None
    assert r_npm.results[0].title == "sqlite"
    assert r_npm.results[0].popularity == 1122428

    # crates_direct（当 repository 为空时回退 url_template）
    crates_eng = DeclarativeEngine(specs["crates_direct"], env={})
    crates_mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "crates": [
                    {
                        "id": "rusqlite",
                        "name": "rusqlite",
                        "description": "Ergonomic wrapper for SQLite",
                        "updated_at": "2026-10-07T00:37:37Z",
                        "downloads": 9876543,
                        "repository": None,
                        "homepage": None,
                        "documentation": None,
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=crates_mock) as client:
        r_crates = await crates_eng.search("sqlite", limit=2, client=client)
    assert r_crates.failure is None
    assert r_crates.results[0].url == "https://crates.io/crates/rusqlite"
    assert r_crates.results[0].popularity == 9876543


@pytest.mark.asyncio
async def test_declarative_drops_malformed_url_without_hostname():
    """入池前必须剔除无 hostname 的残缺链接（如 SearXNG 漏出的 http:///go/v1/...）并记录 warning。"""
    spec = DeclarativeSpec(
        id="searxng_test",
        domain="web",
        tier="L1",
        protocol="http_json",
        request={"method": "GET", "url": "http://127.0.0.1:8080/search", "params": {"q": "{query}"}},
        response={
            "items_path": "results",
            "output_map": {"title": "title", "url": "url", "snippet": "content"},
        },
    )
    mock = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Broken Redirect", "url": "http:///go/v1/AbCdEf", "content": "bad link"},
                    {"title": "Good Page", "url": "https://docs.example.com/ok", "content": "good link"},
                ]
            },
        )
    )
    eng = DeclarativeEngine(spec, env={})
    async with httpx.AsyncClient(transport=mock) as client:
        env = await eng.search("test", limit=5, client=client)

    assert env.failure is None
    assert [r.title for r in env.results] == ["Good Page"]
    assert any("malformed" in w.lower() or "hostname" in w.lower() for w in env.warnings)
