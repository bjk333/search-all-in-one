"""阶段 6A 红灯测试：融合层低相关噪声降权与源级信噪比权重。"""

from __future__ import annotations

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.pool import fuse_results_rrf


def test_fuse_results_rrf_demotes_zero_overlap_noise_and_prioritizes_primary_repos() -> None:
    env_issues = EngineEnvelope(
        engine="github_issues",
        domain="code",
        results=[
            SearchItem(
                id="iss-1",
                domain="code",
                source="github_issues",
                title="117 AUTONOMOUS STUDIO — ChatGPT × Claude Code",
                url="https://github.com/moshehason/117-LAB-TEST/issues/3",
                snippet="Experimental digital-art project shared workroom.",
            ),
            SearchItem(
                id="iss-2",
                domain="code",
                source="github_issues",
                title="Firecrawl MCP server issue discussion",
                url="https://github.com/org/proj/issues/42",
                snippet="Discussion about firecrawl MCP server integration.",
            ),
        ],
    )
    env_repo = EngineEnvelope(
        engine="github_direct",
        domain="code",
        results=[
            SearchItem(
                id="repo-1",
                domain="code",
                source="github_direct",
                title="firecrawl-mcp-server",
                url="https://github.com/mendableai/firecrawl-mcp-server",
                snippet="Official Firecrawl MCP server for web search and extraction.",
            )
        ],
    )

    ranked = fuse_results_rrf(
        [env_issues, env_repo],
        query="firecrawl MCP server search extraction",
    )

    assert len(ranked) == 3  # 不删结果，保持诚实覆盖
    # 一手仓库必须排第 1
    assert ranked[0].id == "repo-1"
    # 零词面重合的无关 issue 必须沉到最后并标 low_lexical_overlap=True
    assert ranked[-1].id == "iss-1"
    assert ranked[-1].extra.get("low_lexical_overlap") is True
    assert ranked[-1].score < ranked[1].score * 0.6


def test_fuse_results_rrf_does_not_penalize_local_semantic_hits() -> None:
    env_local = EngineEnvelope(
        engine="local_index",
        domain="local",
        results=[
            SearchItem(
                id="loc-1",
                domain="local",
                source="local_index",
                title="架构设计笔记",
                url="file:///D:/notes/arch.md",
                snippet="向量语义召回文档，无英文词面重合。",
                extra={"retrieval_mode": "vector"},
            )
        ],
    )
    ranked = fuse_results_rrf([env_local], query="federated metasearch reciprocal rank")
    assert len(ranked) == 1
    assert ranked[0].extra.get("low_lexical_overlap") is not True
