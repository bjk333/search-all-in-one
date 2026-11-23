"""阶段 16 单测：中英双语分词 CamelCase/PascalCase 拆分召回、
local 知识库与 local_code 统一 `path:` 子路径过滤（含防路径穿越）。"""

from __future__ import annotations

import pathlib

import pytest

from omni_search.local_code import LocalCodeEngine, _parse_local_code_query
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.local_query import tokenize_bilingual


def test_tokenize_bilingual_splits_camelcase_and_acronyms():
    """tokenize_bilingual 应保留原词并拆分驼峰/缩写标识符，提升代码与技术文档召回。"""
    toks1 = set(tokenize_bilingual("class FastFailCircuitBreaker:").split())
    assert "fastfailcircuitbreaker" in toks1
    assert {"fast", "fail", "circuit", "breaker"}.issubset(toks1)

    toks2 = set(tokenize_bilingual("OmniMCPServer HTTPResponseParser").split())
    assert "omnimcpserver" in toks2
    assert {"omni", "mcp", "server", "http", "response", "parser"}.issubset(toks2)


@pytest.mark.asyncio
async def test_local_index_recalls_camelcase_symbol_by_natural_words(tmp_path: pathlib.Path):
    """搜自然语言词 'circuit breaker' 应能召回仅含驼峰符号 FastFailCircuitBreaker 的代码文件。"""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "cb.py").write_text(
        "class FastFailCircuitBreaker:\n    def trip(self):\n        return True\n",
        encoding="utf-8",
    )
    (src / "other.py").write_text("def unrelated_helper():\n    return 42\n", encoding="utf-8")

    db = tmp_path / "idx.db"
    index_paths([src], db_path=db)

    eng = LocalIndexEngine(db_path=db)
    env = await eng.search("circuit breaker", limit=5)
    assert env.failure is None
    assert len(env.results) >= 1
    assert env.results[0].title == "cb.py"


@pytest.mark.asyncio
async def test_local_index_path_filter_isolates_subdirectory(tmp_path: pathlib.Path):
    """local 知识库支持 `path:` 子路径精准过滤，仅返回目标子目录下的匹配文档。"""
    alpha = tmp_path / "workspace" / "sub_alpha"
    beta = tmp_path / "workspace" / "sub_beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "notes.md").write_text("# Hybrid RAG\nSQLite hybrid retrieval notes in alpha.\n", encoding="utf-8")
    (beta / "notes.md").write_text("# Hybrid RAG\nSQLite hybrid retrieval notes in beta.\n", encoding="utf-8")

    db = tmp_path / "idx_path.db"
    index_paths([tmp_path / "workspace"], db_path=db)

    eng = LocalIndexEngine(db_path=db)
    env = await eng.search("path:sub_alpha hybrid retrieval", limit=5)
    assert env.failure is None
    assert len(env.results) == 1
    assert "sub_alpha" in env.results[0].url.lower()

    # 仅带 path: 过滤器不含查询词也应列出该子目录下文件
    env_only = await eng.search("path:sub_beta", limit=5)
    assert env_only.failure is None
    assert len(env_only.results) == 1
    assert "sub_beta" in env_only.results[0].url.lower()


def test_parse_local_code_query_supports_path_and_blocks_traversal():
    """_parse_local_code_query 支持 path: 子路径提取并忽略危险 `..` 路径穿越。"""
    pattern, is_ast, globs, paths = _parse_local_code_query("path:src/omni_search lang:py safe_fetch_url")
    assert pattern == "safe_fetch_url"
    assert is_ast is False
    assert "*.py" in globs
    assert paths == ["src/omni_search"]

    p2, _, _, paths2 = _parse_local_code_query("path:../../etc/passwd secret_key")
    assert p2 == "secret_key"
    assert paths2 == []


@pytest.mark.asyncio
async def test_local_code_engine_path_filter_isolates_subdirectory(tmp_path: pathlib.Path):
    """LocalCodeEngine 使用 `path:` 仅在目标子目录中检索匹配代码。"""
    alpha = tmp_path / "pkg_a"
    beta = tmp_path / "pkg_b"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "mod.py").write_text("def target_unique_symbol():\n    return 'a'\n", encoding="utf-8")
    (beta / "mod.py").write_text("def target_unique_symbol():\n    return 'b'\n", encoding="utf-8")

    eng = LocalCodeEngine(roots=[tmp_path])
    env = await eng.search("path:pkg_a lang:py target_unique_symbol", limit=5)
    assert env.failure is None
    assert len(env.results) == 1
    assert "pkg_a" in env.results[0].url.lower()
