"""阶段 10A–10D 回归测试：research_audit 解耦 + v1.0.0 稳定版发版契约校验。"""

from __future__ import annotations

import pathlib
import tomllib

import omni_search
from omni_search import research, research_audit
from omni_search.mcp_server import _TOOLS_SPEC, build_default_engines


_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_research_audit_module_reexports() -> None:
    assert research.classify_evidence_tier is research_audit.classify_evidence_tier
    assert research.compute_confidence is research_audit.compute_confidence
    assert research.extract_pub_year is research_audit.extract_pub_year
    assert research.extract_relevant_quotes is research_audit.extract_relevant_quotes
    assert research.audit_freshness_and_conflicts is research_audit.audit_freshness_and_conflicts
    assert research.audit_verified_citations is research_audit.audit_verified_citations


def test_v1_release_contracts_version_deps_tools_and_engines() -> None:
    pyproj = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = pyproj["project"]

    assert omni_search.__version__ == proj["version"]
    assert tuple(int(p) for p in omni_search.__version__.split(".")[:2]) >= (1, 0)
    # 严守精简依赖契约：仅 4 个核心运行时依赖
    assert len(proj["dependencies"]) == 4

    # 严守 8 个正交 MCP 工具契约
    tool_names = [t["name"] for t in _TOOLS_SPEC]
    assert tool_names == [
        "omni_search",
        "omni_fetch",
        "omni_batch",
        "omni_research",
        "omni_index",
        "omni_doctor",
        "omni_monitor",
        "omni_verify",
    ]

    # 50 个原生引擎覆盖 11 大能力域
    engines = build_default_engines(env={})
    assert len(engines) == 50
    assert len({e.domain for e in engines}) == 11


def test_v1_docs_sync_with_latest_cli_features() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for doc in (readme, skill):
        assert "--reset-engines" in doc
        assert "--no-relax" in doc
        assert "omni verify" in doc
        assert "omni crawl" in doc
        assert "untrusted_content=true" in doc
