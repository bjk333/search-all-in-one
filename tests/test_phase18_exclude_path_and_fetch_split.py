"""Phase 18: local/local_code 统一 -path: 排除子目录过滤 + fetch_apis 模块瘦身守卫。"""

from __future__ import annotations

import pathlib

import pytest

from omni_search.local_chunking import _path_matches_filters, parse_local_query_full
from omni_search.local_code import LocalCodeEngine, _parse_local_code_query
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.router import route_domains


def test_parse_local_query_full_supports_exclude_path_and_blocks_traversal() -> None:
    cleaned, mods, exts, cols, tags, req_paths, exc_paths = parse_local_query_full(
        "path:src -path:tests,legacy -path:../../etc/passwd -path:/abs/dir type:code circuit breaker"
    )
    assert cleaned == "circuit breaker"
    assert mods == {"code"}
    assert req_paths == {"src"}
    assert exc_paths == {"tests", "legacy"}

    # 匹配：在 src 下且不在 tests/legacy 下 -> 通过
    assert _path_matches_filters(
        "D:/repo/src/omni_search/pool.py",
        "code",
        mods,
        exts,
        required_paths=req_paths,
        excluded_paths=exc_paths,
    ) is True
    # 匹配：在 src/tests 下 -> 被 -path:tests 排除
    assert _path_matches_filters(
        "D:/repo/src/tests/test_pool.py",
        "code",
        mods,
        exts,
        required_paths=req_paths,
        excluded_paths=exc_paths,
    ) is False


@pytest.mark.asyncio
async def test_local_kb_search_excludes_subpath(tmp_path: pathlib.Path) -> None:
    prod_dir = tmp_path / "src"
    test_dir = tmp_path / "tests"
    prod_dir.mkdir()
    test_dir.mkdir()
    (prod_dir / "engine.md").write_text("Hybrid retrieval circuit breaker architecture guide", encoding="utf-8")
    (test_dir / "test_engine.md").write_text("Hybrid retrieval circuit breaker mock test fixture", encoding="utf-8")

    db = tmp_path / "idx.db"
    index_paths([tmp_path], db_path=db)

    eng = LocalIndexEngine(db_path=db)
    env = await eng.search("-path:tests circuit breaker", limit=5)
    assert env.failure is None
    assert len(env.results) == 1
    assert "engine.md" in env.results[0].url
    assert "test_engine.md" not in env.results[0].url


@pytest.mark.asyncio
async def test_local_code_supports_exclude_path_filter(tmp_path: pathlib.Path) -> None:
    pattern, is_ast, globs, paths = _parse_local_code_query(
        "path:src -path:tests -path:../../etc lang:py FastFailCircuitBreaker"
    )
    assert pattern == "FastFailCircuitBreaker"
    assert is_ast is False
    assert paths == ["src"]
    assert "*.py" in globs
    assert "!tests/**" in globs
    assert "!**/tests/**" in globs
    assert all("etc" not in g for g in globs)

    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "breaker.py").write_text("class FastFailCircuitBreaker:\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_breaker.py").write_text("class FastFailCircuitBreaker:\n    pass\n", encoding="utf-8")

    eng = LocalCodeEngine(roots=[tmp_path])
    ok, _ = eng.credential_status()
    if not ok:
        pytest.skip("rg not available")
    env = await eng.search("-path:tests FastFailCircuitBreaker", limit=10)
    assert env.failure is None
    assert len(env.results) >= 1
    assert all("test_breaker.py" not in it.url for it in env.results)
    assert any("breaker.py" in it.url for it in env.results)


def test_router_recognizes_exclude_path_filter() -> None:
    domains = route_domains("-path:tests circuit breaker", ["auto"])
    assert "local" in domains
    assert "local_code" in domains


def test_fetch_apis_modular_split_and_line_budget() -> None:
    import omni_search.fetch_apis as fa
    import omni_search.fetch_apis_extended as fa_ext

    assert callable(fa._try_fetch_official_api)
    assert callable(fa_ext._try_fetch_extended_official_api)

    src_dir = pathlib.Path(fa.__file__).resolve().parent
    fa_lines = len((src_dir / "fetch_apis.py").read_text(encoding="utf-8").splitlines())
    ext_lines = len((src_dir / "fetch_apis_extended.py").read_text(encoding="utf-8").splitlines())
    assert fa_lines < 450, f"fetch_apis.py too long: {fa_lines}"
    assert ext_lines < 450, f"fetch_apis_extended.py too long: {ext_lines}"

    max_lines = max(
        len(p.read_text(encoding="utf-8").splitlines())
        for p in src_dir.glob("*.py")
    )
    assert max_lines < 650, f"repo max module lines exceeded 650: {max_lines}"
