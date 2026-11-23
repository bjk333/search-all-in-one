"""Phase 3C 红灯测试：零 LLM Contextual Chunking 层级面包屑 + news/security 温和时效新鲜度衰减。"""

from __future__ import annotations

import pytest

from omni_search.contracts import EngineEnvelope, SearchItem
from omni_search.local_index import LocalIndexEngine, _split_line_chunks, index_paths
from omni_search.pool import fuse_results_rrf


@pytest.mark.asyncio
async def test_contextual_chunking_injects_markdown_headings_and_class_breadcrumbs(tmp_path):
    """深处 Markdown 与 Python 分块应自动注入上级标题/类面包屑上下文，提升深处片段召回。"""
    md_lines = [
        "# Storage Architecture",
        "",
        "## Circuit Breaker",
        "",
    ]
    for i in range(42):
        md_lines.append(f"Routine operational line {i} describing normal throughput telemetry.")
    md_lines.append("When consecutive timeout retries exceed four threshold it opens immediately.")
    md_text = "\n".join(md_lines) + "\n"

    chunks = _split_line_chunks(md_text)
    assert len(chunks) >= 2
    last_loc, last_body = chunks[-1]
    assert "Storage Architecture" in last_body
    assert "Circuit Breaker" in last_body
    assert "exceed four threshold" in last_body

    py_lines = ["class PaymentGateway:"]
    for i in range(42):
        py_lines.append(f"    def _step_{i}(self):\n        return {i}")
    py_lines.append("    def refund_settlement(self, tx_id: str) -> bool:\n        return bool(tx_id)")
    py_text = "\n".join(py_lines) + "\n"

    py_chunks = _split_line_chunks(py_text)
    assert any("class PaymentGateway" in body and "refund_settlement" in body for _, body in py_chunks)

    # 验证 Contextual Breadcrumb 数学上严格提升深处片段与“标题+正文”联合查询的向量相似度
    from omni_search.embeddings import HashNgramEmbedder, dot_similarity

    emb = HashNgramEmbedder()
    raw_without_ctx = "\n".join(line for line in last_body.splitlines() if not line.startswith("[ctx:"))
    q_vec, v_with, v_without = emb.embed_texts(
        ["Storage Architecture Circuit Breaker exceed four threshold", last_body, raw_without_ctx]
    )
    assert dot_similarity(q_vec, v_with) > dot_similarity(q_vec, v_without)

    ws = tmp_path / "ctx_ws"
    ws.mkdir()
    (ws / "handbook.md").write_text(md_text, encoding="utf-8")
    (ws / "gateway.py").write_text(py_text, encoding="utf-8")
    db = tmp_path / "ctx.db"
    index_paths([ws], db_path=db)

    eng = LocalIndexEngine(db_path=db)
    r_md = await eng.search("Circuit Breaker exceed four threshold", limit=3)
    assert r_md.results and r_md.results[0].title == "handbook.md"
    assert r_md.results[0].extra.get("matched_locator") == last_loc

    r_py = await eng.search("PaymentGateway refund_settlement", limit=3)
    assert r_py.results and r_py.results[0].title == "gateway.py"


def test_recency_decay_prioritizes_recent_news_and_security_without_penalizing_knowledge():
    """news/security 域应优先近期条目，knowledge 域经典权威条目不受时效惩罚。"""
    old_news = EngineEnvelope(
        engine="src_a",
        domain="news",
        results=[
            SearchItem(
                id="n_old",
                domain="news",
                source="src_a",
                title="SQLite Hybrid Search Release Notes",
                url="https://old.example.org/sqlite-2015",
                snippet="SQLite hybrid search news coverage.",
                published_at="2015-05-01T00:00:00Z",
            )
        ],
    )
    new_news = EngineEnvelope(
        engine="src_b",
        domain="news",
        results=[
            SearchItem(
                id="n_new",
                domain="news",
                source="src_b",
                title="SQLite Hybrid Search Release Notes",
                url="https://new.example.org/sqlite-2026",
                snippet="SQLite hybrid search news coverage.",
                published_at="2026-10-15T00:00:00Z",
            )
        ],
    )
    fused_news = fuse_results_rrf([old_news, new_news], query="SQLite Hybrid Search")
    assert fused_news[0].id == "n_new"
    assert fused_news[1].id == "n_old"

    old_sec = EngineEnvelope(
        engine="nvd_cve",
        domain="security",
        results=[
            SearchItem(
                id="s_old",
                domain="security",
                source="nvd_cve",
                title="OpenSSL TLS vulnerability",
                url="https://nvd.nist.gov/vuln/detail/CVE-2016-0001",
                snippet="OpenSSL TLS memory issue.",
                published_at="2016-02-01T00:00:00Z",
            )
        ],
    )
    new_sec = EngineEnvelope(
        engine="osv_vulns",
        domain="security",
        results=[
            SearchItem(
                id="s_new",
                domain="security",
                source="osv_vulns",
                title="OpenSSL TLS vulnerability",
                url="https://osv.dev/vulnerability/CVE-2026-0001",
                snippet="OpenSSL TLS memory issue.",
                published_at="2026-11-01T00:00:00Z",
            )
        ],
    )
    fused_sec = fuse_results_rrf([old_sec, new_sec], query="OpenSSL TLS vulnerability")
    assert fused_sec[0].id == "s_new"

    # knowledge 域不惩罚经典高排名条目
    know_env = EngineEnvelope(
        engine="wikipedia",
        domain="knowledge",
        results=[
            SearchItem(
                id="k_classic",
                domain="knowledge",
                source="wikipedia",
                title="Reciprocal Rank Fusion",
                url="https://en.wikipedia.org/wiki/RRF",
                snippet="Classic IR fusion method.",
                published_at="2012-01-01T00:00:00Z",
            ),
            SearchItem(
                id="k_newer",
                domain="knowledge",
                source="wikipedia",
                title="Reciprocal Rank Fusion",
                url="https://en.wikipedia.org/wiki/RRF_2026",
                snippet="Classic IR fusion method.",
                published_at="2026-01-01T00:00:00Z",
            ),
        ],
    )
    fused_know = fuse_results_rrf([know_env], query="Reciprocal Rank Fusion")
    assert fused_know[0].id == "k_classic"
