"""Phase 17: verifier 硬事实锚点边界防幻觉 + router local_code/过滤器自动路由测试。"""

from __future__ import annotations

import pytest

from omni_search.fetcher import FetchResult
from omni_search.router import route_domains
from omni_search.verifier import _evaluate_claim_against_page, _token_in_text


def test_token_in_text_prevents_version_cve_number_and_acronym_false_positives() -> None:
    # 版本号不能被更长版本或前导数字子串误命中
    assert _token_in_text("v2.0", "released v2.0.5 today") is False
    assert _token_in_text("3.1", "supports python 3.13 and 13.10") is False
    assert _token_in_text("v2.0.5", "released 2.0.5 today") is True
    assert _token_in_text("2.0.5", "released v2.0.5 today") is True
    assert _token_in_text("v1.6.0", "search-all-in-one (v1.6.0).") is True

    # CVE 编号不能前缀误命中
    assert _token_in_text("CVE-2024-1234", "patched cve-2024-12345 in latest build") is False
    assert _token_in_text("CVE-2024-1234", "patched cve-2024-1234 in latest build") is True

    # 数字与百分比边界 + 千分位兼容
    assert _token_in_text("2025", "port 20250 is open") is False
    assert _token_in_text("50%", "achieved 150% speedup") is False
    assert _token_in_text("50%", "achieved 50% speedup") is True
    assert _token_in_text("1000", "indexed 1,000 documents") is True
    assert _token_in_text("1,000", "indexed 1000 documents") is True

    # 缩写/英文词边界，避免子串假阳性
    assert _token_in_text("RAG", "memory fragmentation issue") is False
    assert _token_in_text("MCP", "mcpack archive format") is False
    assert _token_in_text("RAG", "hybrid rag pipeline") is True

    # 中文仍保持正常子串匹配
    assert _token_in_text("检索", "本地知识库检索精度提升") is True


def test_evaluate_claim_catches_version_and_cve_hallucinations() -> None:
    page = FetchResult(
        ok=True,
        url="https://example.com/release",
        title="Release Notes",
        content="Python 3.13 fixes CVE-2024-12345 and improves memory fragmentation by 150%.",
        status_code=200,
    )
    # 声称错误版本号 3.1 与错误 CVE-2024-1234 与不存在的 RAG
    item = _evaluate_claim_against_page(
        "Python 3.1 fixes CVE-2024-1234 for RAG pipelines",
        page.url,
        page,
    )
    assert "3.1" in item.missing_terms
    assert "CVE-2024-1234" in item.missing_terms
    assert "RAG" in item.missing_terms
    assert item.anchor_coverage == 0.0
    assert item.verdict != "supported"


def test_router_auto_detects_local_code_and_explicit_filters() -> None:
    # 本地代码显式中文/英文意图 -> 路由到 local_code
    d1 = route_domains("在本地源码里查找 FastFailCircuitBreaker", ["auto"])
    assert "local_code" in d1
    assert "local" in d1

    d2 = route_domains("search local code for route_domains", ["auto"])
    assert "local_code" in d2

    # 统一过滤器 path: -> 同时覆盖 local 与 local_code
    d3 = route_domains("path:src/omni_search circuit breaker", ["auto"])
    assert "local" in d3
    assert "local_code" in d3

    # 本机代码专属过滤器 symbol: / ast: -> 路由到 local_code
    d4 = route_domains("symbol:LocalCodeEngine", ["auto"])
    assert "local_code" in d4

    # 本地知识库过滤器 type: / ext: -> 路由到 local
    d5 = route_domains("type:pdf quarterly report", ["auto"])
    assert "local" in d5
    d6 = route_domains("ext:md architecture notes", ["auto"])
    assert "local" in d6

    # 年份新近性动态覆盖 2027
    d7 = route_domains("ai framework roadmap 2027", ["auto"])
    assert "news" in d7
