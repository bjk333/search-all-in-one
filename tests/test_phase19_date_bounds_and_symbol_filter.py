"""Phase 19: 本地知识库 after:/before: 时间过滤 + local_code symbol: 跨语言符号定义精准查询。"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import pathlib

import pytest

from omni_search.local_chunking import _path_matches_filters, parse_local_date_bounds
from omni_search.local_code import LocalCodeEngine, _parse_local_code_query
from omni_search.local_index import LocalIndexEngine, index_paths


def test_parse_local_date_bounds_and_mtime_matching() -> None:
    cleaned, after_ns, before_ns = parse_local_date_bounds(
        "after:2025-06-01 before:2026-01-31 after:invalid-date hybrid rag"
    )
    assert cleaned == "hybrid rag"
    assert after_ns is not None and before_ns is not None
    assert after_ns < before_ns

    t_2024 = int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    t_2025_10 = int(datetime(2025, 10, 15, tzinfo=timezone.utc).timestamp() * 1e9)
    t_2026_03 = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1e9)

    assert _path_matches_filters("a.md", "document", set(), set(), mtime_ns=t_2025_10, after_ns=after_ns, before_ns=before_ns) is True
    assert _path_matches_filters("a.md", "document", set(), set(), mtime_ns=t_2024, after_ns=after_ns, before_ns=before_ns) is False
    assert _path_matches_filters("a.md", "document", set(), set(), mtime_ns=t_2026_03, after_ns=after_ns, before_ns=before_ns) is False


@pytest.mark.asyncio
async def test_local_index_filters_by_after_and_before_dates(tmp_path: pathlib.Path) -> None:
    docs = tmp_path / "notes"
    docs.mkdir()
    old_f = docs / "old_note.md"
    new_f = docs / "new_note.md"
    old_f.write_text("Hybrid retrieval circuit breaker architecture notes", encoding="utf-8")
    new_f.write_text("Hybrid retrieval circuit breaker architecture notes", encoding="utf-8")

    ts_old = datetime(2024, 3, 10, tzinfo=timezone.utc).timestamp()
    ts_new = datetime(2026, 2, 20, tzinfo=timezone.utc).timestamp()
    os.utime(old_f, (ts_old, ts_old))
    os.utime(new_f, (ts_new, ts_new))

    db = tmp_path / "idx.db"
    index_paths([docs], db_path=db)
    eng = LocalIndexEngine(db_path=db)

    env_after = await eng.search("after:2025-01-01 circuit breaker", limit=5)
    assert env_after.failure is None
    assert len(env_after.results) == 1
    assert "new_note.md" in env_after.results[0].url

    env_before = await eng.search("before:2024-12-31 circuit breaker", limit=5)
    assert env_before.failure is None
    assert len(env_before.results) == 1
    assert "old_note.md" in env_before.results[0].url


@pytest.mark.asyncio
async def test_local_code_symbol_filter_matches_definitions_only(tmp_path: pathlib.Path) -> None:
    pat, is_ast, globs, paths = _parse_local_code_query("lang:py path:src symbol:FastFailCircuitBreaker")
    assert is_ast is False
    assert "*.py" in globs
    assert paths == ["src"]
    assert "FastFailCircuitBreaker" in pat
    assert "class" in pat and "def" in pat

    # 非法符号字符被安全拒绝（防止正则注入）
    bad_pat, _, _, _ = _parse_local_code_query("symbol:Foo.*Bar")
    assert ".*" not in bad_pat

    src = tmp_path / "src"
    src.mkdir()
    (src / "definition.py").write_text("class FastFailCircuitBreaker:\n    pass\n", encoding="utf-8")
    (src / "caller.py").write_text("cb = FastFailCircuitBreaker()\n", encoding="utf-8")

    eng = LocalCodeEngine(roots=[tmp_path])
    ok, _ = eng.credential_status()
    if not ok:
        pytest.skip("rg not available")

    env = await eng.search("lang:py symbol:FastFailCircuitBreaker", limit=10)
    assert env.failure is None
    assert len(env.results) == 1
    assert "definition.py" in env.results[0].url
    assert all("caller.py" not in r.url for r in env.results)
