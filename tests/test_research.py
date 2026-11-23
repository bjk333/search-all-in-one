import httpx
import pytest

from omni_search.contracts import EngineEnvelope, EngineFailure, ReasonCode, SearchItem
from omni_search.fetcher import FetchResult
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool
from omni_search.research import (
    classify_evidence_tier,
    plan_research_lanes,
    render_research_markdown,
    run_research_rounds,
    widen_query,
)


class _MockEngine:
    def __init__(self, eid: str, domain: str, fn):
        self.id = eid
        self.domain = domain
        self.tier = "L0"
        self.deadline_sec = 5.0
        self._fn = fn

    async def search(self, query: str, limit: int, client) -> EngineEnvelope:
        return self._fn(query, limit)


def test_widen_query_extracts_core_terms_from_long_brief():
    w = widen_query("how to build a federated multimodal search engine for ai agents in 2026")
    parts = w.split()
    assert 2 <= len(parts) <= 4
    assert "how" not in parts and "to" not in parts


def test_plan_research_lanes_generates_deterministic_plan_id_and_lanes():
    p1 = plan_research_lanes("federated search architecture")
    p2 = plan_research_lanes("federated search architecture")
    p3 = plan_research_lanes("different topic")
    assert p1.plan_id == p2.plan_id
    assert p1.plan_id != p3.plan_id
    assert len(p1.lanes) >= 3
    names = {ln.name for ln in p1.lanes}
    assert {"overview", "academic", "community"}.issubset(names)


@pytest.mark.asyncio
async def test_run_research_rounds_executes_discover_verify_and_gap_fill(tmp_path):
    calls: list[tuple[str, str]] = []

    def wiki_fn(q: str, limit: int) -> EngineEnvelope:
        calls.append(("wiki", q))
        return EngineEnvelope(
            engine="wiki",
            domain="knowledge",
            results=[
                SearchItem(id="1", domain="knowledge", source="wiki", title="Metasearch", url="https://docs.example.com/meta", snippet="Overview"),
            ],
        )

    def acad_fn(q: str, limit: int) -> EngineEnvelope:
        calls.append(("acad", q))
        # 长查询首次 0 命中，宽词补查才命中
        if len(q.split()) > 3:
            return EngineEnvelope(engine="acad", domain="academic", results=[])
        return EngineEnvelope(
            engine="acad",
            domain="academic",
            results=[
                SearchItem(id="2", domain="academic", source="acad", title="Metasearch Paper", url="https://arxiv.example.com/abs/1234", snippet="Paper"),
                SearchItem(id="1b", domain="academic", source="acad", title="Metasearch", url="https://docs.example.com/meta", snippet="Cross-lane hit"),
            ],
        )

    def news_fn(q: str, limit: int) -> EngineEnvelope:
        calls.append(("news", q))
        return EngineEnvelope(
            engine="news",
            domain="news",
            results=[],
            failure=EngineFailure(reason_code=ReasonCode.RATE_LIMITED, reason="HTTP 429"),
        )

    pool = FederatedPool(
        engines=[
            _MockEngine("wiki", "knowledge", wiki_fn),
            _MockEngine("acad", "academic", acad_fn),
            _MockEngine("news", "news", news_fn),
        ],
        health_store=HealthStore(tmp_path / "h.json"),
    )

    async def fake_fetch(url: str, max_chars: int = 4000, timeout_sec: float = 12.0) -> FetchResult:
        if "docs.example.com" in url:
            return FetchResult(
                ok=True,
                url=url,
                final_url=url,
                status_code=200,
                title="Metasearch Official",
                content="Verified authoritative content.",
                untrusted_content=True,
            )
        return FetchResult(ok=False, url=url, reason_code="http-error", reason="HTTP 403")

    report = await run_research_rounds(
        pool=pool,
        brief="how to design federated metasearch architecture for agents",
        domains=["knowledge", "academic", "news"],
        limit=3,
        verify_top_k=2,
        budget_ms=10000,
        fetcher=fake_fetch,
    )

    assert report.plan_id
    # Round 1 + Round 3 缺口补查：academic 在 Round 1 0 命中后应被宽词补查命中
    assert len(report.gap_fills) >= 1
    assert any(c.url == "https://arxiv.example.com/abs/1234" for c in report.candidates)

    # 跨 lane 交叉命中的候选应排首位且记录 matched_lanes
    top = report.candidates[0]
    assert top.url == "https://docs.example.com/meta"
    assert len(top.extra.get("matched_lanes", [])) >= 2

    # Round 2 原文核验
    assert len(report.verified_evidence) >= 1
    verified_ok = [v for v in report.verified_evidence if v.verified]
    assert len(verified_ok) == 1
    assert verified_ok[0].untrusted_content is True
    assert "Verified authoritative" in verified_ok[0].excerpt

    # 缺口审计应显式披露 news 的 rate-limited 与失败的 fetch
    assert any("rate-limited" in g or "news" in g for g in report.gaps)


def test_research_modes_generate_distinct_lanes_and_plan_ids():
    p_std = plan_research_lanes("universal search engine", mode="standard")
    p_feas = plan_research_lanes("universal search engine", mode="feasibility", extra_terms=["metasearch", "mcp"])
    p_bench = plan_research_lanes("universal search engine", mode="benchmark")
    p_fact = plan_research_lanes("universal search engine", mode="factcheck")

    assert len({p_std.plan_id, p_feas.plan_id, p_bench.plan_id, p_fact.plan_id}) == 4
    feas_names = {ln.name for ln in p_feas.lanes}
    assert {"overview", "alternatives", "risks", "academic"}.issubset(feas_names)
    # 不同 lane 查询应有真实差异，且融入 extra_terms
    assert len({ln.query for ln in p_feas.lanes}) >= 3
    assert any("metasearch" in ln.query for ln in p_feas.lanes)


def test_evidence_tier_classification():
    assert classify_evidence_tier("https://arxiv.org/abs/2601.02365", "arxiv") == "T1"
    assert classify_evidence_tier("https://docs.python.org/3/library/sqlite3.html", "searxng") == "T1"
    assert classify_evidence_tier("https://github.com/searxng/searxng", "github_direct") == "T1"
    assert classify_evidence_tier("https://en.wikipedia.org/wiki/Metasearch_engine", "wikipedia") == "T2"
    assert classify_evidence_tier("https://someblog.example.com/post", "searxng") == "T3"


@pytest.mark.asyncio
async def test_research_diversifies_host_verification_and_renders_markdown(tmp_path):
    # 同一主机返回 2 条高分 + 另一主机返回 1 条
    def wiki_fn(q: str, limit: int) -> EngineEnvelope:
        return EngineEnvelope(
            engine="wiki",
            domain="knowledge",
            results=[
                SearchItem(id="1", domain="knowledge", source="wiki", title="Wiki A", url="https://en.wikipedia.org/wiki/A"),
                SearchItem(id="2", domain="knowledge", source="wiki", title="Wiki B", url="https://en.wikipedia.org/wiki/B"),
            ],
        )

    def acad_fn(q: str, limit: int) -> EngineEnvelope:
        return EngineEnvelope(
            engine="arxiv",
            domain="academic",
            results=[
                SearchItem(id="3", domain="academic", source="arxiv", title="Arxiv Paper", url="https://arxiv.org/abs/2601.00001"),
            ],
        )

    pool = FederatedPool(
        engines=[_MockEngine("wiki", "knowledge", wiki_fn), _MockEngine("arxiv", "academic", acad_fn)],
        health_store=HealthStore(tmp_path / "h.json"),
    )
    fetched_urls: list[str] = []

    async def fake_fetch(url: str, max_chars: int = 2000, timeout_sec: float = 10.0) -> FetchResult:
        fetched_urls.append(url)
        return FetchResult(ok=True, url=url, status_code=200, title=f"Title for {url}", content="Clean excerpt.", untrusted_content=True)

    report = await run_research_rounds(
        pool=pool,
        brief="metasearch architecture",
        mode="feasibility",
        domains=["knowledge", "academic"],
        limit=3,
        verify_top_k=2,
        fetcher=fake_fetch,
    )

    # Top-2 核验应覆盖 2 个不同域名（而非重复抓两个 wikipedia）
    assert len(fetched_urls) == 2
    assert any("arxiv.org" in u for u in fetched_urls)
    assert any("wikipedia.org" in u for u in fetched_urls)

    # 候选应带 evidence_tier 与 confidence
    for c in report.candidates:
        assert c.extra.get("evidence_tier") in ("T1", "T2", "T3")
        assert c.extra.get("confidence") in ("high", "medium", "low")

    md = render_research_markdown(report)
    assert report.plan_id in md
    assert "feasibility" in md
    assert "覆盖率" in md
    assert "T1" in md
    assert "untrusted_content" in md or "不可信外部数据" in md


def test_render_markdown_sequential_headings_without_merge():
    """无多 Agent 合单审计时，Markdown 章节号必须连续（一..六），不得从四跳到六。"""
    from omni_search.research import ResearchReport

    rep = ResearchReport(version="0.1.0", plan_id="abc123456789", brief="test sequential", mode="standard", lanes=[], candidates=[])
    md = render_research_markdown(rep)
    assert "## 四、时效与冲突审计" in md
    assert "## 五、覆盖缺口与未验证项" in md
    assert "## 六、复现命令" in md
    assert "## 七、" not in md


def test_extract_quotes_skips_spaceless_tag_blob():
    """Quote Grounding 应过滤 GitHub topics 等无空格长粘连标签串，优先抽取正常自然语言证据句。"""
    from omni_search.research import extract_relevant_quotes

    blob = (
        "ai-agent-memoryai-agentsbunclaude-codeembeddingshermeshybrid-searchllama-cpplocal-firstmcp-serverragvector-searchsqlite\n\n"
        "This library combines SQLite FTS5 BM25 lexical search with dense vector search via reciprocal rank fusion."
    )
    quotes = extract_relevant_quotes(blob, "sqlite hybrid vector search", max_quotes=2)
    assert len(quotes) == 1
    assert "reciprocal rank fusion" in quotes[0]
    assert "ai-agent-memoryai-agents" not in quotes[0]


def test_render_citations_bibtex_generates_clean_entries():
    """BibTeX 导出应生成唯一 citekey、识别 arXiv/DOI、转义特殊字符并标注证据等级。"""
    from omni_search.research import ResearchReport, VerifiedEvidence, render_citations_bibtex

    rep = ResearchReport(
        version="0.1.0",
        plan_id="bib123456789",
        brief="sqlite hybrid search",
        mode="benchmark",
        lanes=[],
        candidates=[
            SearchItem(
                id="c1",
                domain="academic",
                source="arxiv",
                title="SQLite is Enough: 100% Local & Fast",
                url="https://arxiv.org/abs/2608.24060v1",
                published_at="2026-08-20",
                extra={"evidence_tier": "T1", "pub_year": 2026},
            ),
            SearchItem(
                id="c2",
                domain="academic",
                source="crossref",
                title="Hybrid Retrieval_Bench",
                url="https://doi.org/10.1007/978-1-4302-0172-4_9",
                published_at="2025-05-01",
                extra={"evidence_tier": "T1", "pub_year": 2025},
            ),
            SearchItem(
                id="c3",
                domain="code",
                source="github_direct",
                title="yoloshii/ClawMem",
                url="https://github.com/yoloshii/ClawMem",
                published_at="2026-09-01",
                extra={"evidence_tier": "T1", "pub_year": 2026},
            ),
        ],
        verified_evidence=[
            VerifiedEvidence(
                url="https://arxiv.org/abs/2608.24060v1",
                title="SQLite is Enough: 100% Local & Fast",
                domain="academic",
                source="arxiv",
                evidence_tier="T1",
                verified=True,
                status_code=200,
            )
        ],
    )
    bib = render_citations_bibtex(rep)
    assert "@article{" in bib
    assert "eprint = {2608.24060}" in bib
    assert "archivePrefix = {arXiv}" in bib
    assert "doi = {10.1007/978-1-4302-0172-4_9}" in bib
    assert "@misc{" in bib
    assert r"100\% Local \& Fast" in bib
    assert "verified=yes" in bib


def test_extract_followup_terms_picks_cross_candidate_resonant_terms():
    """确定性自反提词应选出在 >=2 候选中共振且不在原 brief 中的专业术语。"""
    from omni_search.research import extract_followup_terms

    cands = [
        SearchItem(
            id="1",
            domain="academic",
            source="arxiv",
            title="scrydb: Hybrid Search with sqlite-vec and BM25",
            url="https://arxiv.org/abs/2608.24060",
            snippet="Combines FTS5 BM25 with sqlite-vec dense vectors.",
        ),
        SearchItem(
            id="2",
            domain="code",
            source="github_direct",
            title="asg017/sqlite-vec",
            url="https://github.com/asg017/sqlite-vec",
            snippet="A vector search SQLite extension paired with BM25 lexical ranking.",
        ),
        SearchItem(
            id="3",
            domain="web",
            source="searxng",
            title="Random one-off noisewordxyz article",
            url="https://example.com/noise",
            snippet="Mentions uniqueonlytoken once.",
        ),
    ]
    terms = extract_followup_terms(cands, brief="sqlite hybrid search", max_terms=2)
    assert "sqlite-vec" in terms
    assert "bm25" in terms
    assert "sqlite" not in terms
    assert "noisewordxyz" not in terms


@pytest.mark.asyncio
async def test_research_include_local_and_deepen_round(tmp_path):
    """联合调研 include_local=True 应纳入 local_kb、跳过 file:// 核验与 BibTeX，且 deepen=True 触发自反追问。"""
    from omni_search.research import render_citations_bibtex

    seen_queries: list[tuple[str, str]] = []

    class _LocalMock:
        id = "local_index"
        domain = "local"
        tier = "L0"
        deadline_sec = 3.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            seen_queries.append(("local_index", query))
            return EngineEnvelope(
                engine="local_index",
                domain="local",
                results=[
                    SearchItem(
                        id="loc1",
                        domain="local",
                        source="local_index",
                        title="local_notes.md",
                        url="file:///D:/private/local_notes.md",
                        snippet="Local note discussing sqlite-vec and bm25.",
                    )
                ],
            )

    class _WebMock:
        id = "arxiv"
        domain = "academic"
        tier = "L0"
        deadline_sec = 5.0

        async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
            seen_queries.append(("arxiv", query))
            if "sqlite-vec" in query.lower() or "bm25" in query.lower():
                return EngineEnvelope(
                    engine="arxiv",
                    domain="academic",
                    results=[
                        SearchItem(
                            id="deep1",
                            domain="academic",
                            source="arxiv",
                            title="Deep Followup: sqlite-vec Benchmarks",
                            url="https://arxiv.org/abs/2609.99999",
                            snippet="Detailed benchmark of sqlite-vec and bm25.",
                            published_at="2026-09-10",
                        )
                    ],
                )
            return EngineEnvelope(
                engine="arxiv",
                domain="academic",
                results=[
                    SearchItem(
                        id="r1",
                        domain="academic",
                        source="arxiv",
                        title="SQLite Hybrid Search using sqlite-vec and BM25",
                        url="https://arxiv.org/abs/2608.24060",
                        snippet="Uses sqlite-vec extension and bm25 scoring.",
                        published_at="2026-08-20",
                    ),
                    SearchItem(
                        id="r2",
                        domain="academic",
                        source="arxiv",
                        title="Evaluating sqlite-vec with BM25 Reranking",
                        url="https://arxiv.org/abs/2608.11111",
                        snippet="Compares sqlite-vec and bm25 latency.",
                        published_at="2026-08-15",
                    ),
                ],
            )

    fetched: list[str] = []

    async def fake_fetch(url: str, max_chars: int = 3500, timeout_sec: float = 12.0) -> FetchResult:
        fetched.append(url)
        return FetchResult(ok=True, url=url, final_url=url, status_code=200, title="Paper", content="Clean paper abstract about sqlite-vec and bm25 hybrid search.")

    pool = FederatedPool(engines=[_LocalMock(), _WebMock()], health_store=HealthStore(tmp_path / "h.json"))
    report = await run_research_rounds(
        pool=pool,
        brief="sqlite hybrid search",
        domains=["academic"],
        include_local=True,
        deepen=True,
        limit=3,
        verify_top_k=2,
        budget_ms=15000,
        render_markdown=True,
        render_bibtex=True,
        fetcher=fake_fetch,
    )

    assert any(ln.name == "local_kb" for ln in report.lanes)
    assert any(gf.name == "followup:deepen" for gf in report.gap_fills)
    assert any("2609.99999" in c.url for c in report.candidates)
    # file:// 绝不得送去公网 fetcher，也不得泄漏进可分享的 BibTeX
    assert all(not u.startswith("file://") for u in fetched)
    assert report.bibtex is not None
    assert "file://" not in report.bibtex
    assert "local_notes.md" not in report.bibtex
    assert "2608.24060" in report.bibtex
    # Markdown 应清晰区分本机已有资产与公网发现
    assert report.markdown is not None
    assert "本机" in report.markdown and "local_notes.md" in report.markdown


def test_widen_query_splits_unspaced_chinese_brief():
    """无空格长中文主题必须能拆分精简，不能原样返回整句 14 字长串。"""
    w = widen_query("大模型智能体记忆系统混合检索", max_terms=3)
    assert w != "大模型智能体记忆系统混合检索"
    assert len(w) < len("大模型智能体记忆系统混合检索")
    assert "大模型" in w and "混合检索" in w


def test_plan_research_lanes_uses_chinese_suffixes_for_cjk_brief():
    """纯中文主题在 benchmark/feasibility 模式下应使用中文后缀，不得拼英文 open source/evaluation 污染英文源。"""
    plan = plan_research_lanes("大模型智能体记忆系统混合检索", mode="benchmark")
    for ln in plan.lanes:
        assert "open source" not in ln.query.lower()
        assert "evaluation" not in ln.query.lower()
        assert "comparison benchmark" not in ln.query.lower()
    assert any("开源" in ln.query for ln in plan.lanes)
    assert any("评测" in ln.query for ln in plan.lanes)


def test_extract_followup_terms_mines_chinese_resonant_ngrams_and_ignores_noise():
    """中文自反提词应挖掘 2-4 字跨文档共振专业术语（如知识图谱/向量），并过滤 published 等泛词。"""
    from omni_search.research import extract_followup_terms

    cands = [
        SearchItem(
            id="z1",
            domain="web",
            source="searxng",
            title="基于知识图谱与向量数据库的大模型智能体长期记忆实践",
            url="https://example.com/1",
            snippet="published 2026: 本文结合知识图谱和向量检索优化智能体记忆。",
        ),
        SearchItem(
            id="z2",
            domain="academic",
            source="openalex",
            title="融合知识图谱的智能体向量检索增强记忆框架",
            url="https://example.com/2",
            snippet="published in 2026: 通过知识图谱提升多跳记忆召回率。",
        ),
    ]
    terms = extract_followup_terms(cands, brief="大模型智能体记忆系统混合检索", max_terms=2)
    assert "published" not in terms
    assert "知识图谱" in terms
    assert any("向量" in t for t in terms)


def test_extract_relevant_quotes_matches_chinese_long_query():
    """中文长无空格查询应通过 2-gram 命中正文证据句，而非返回空列表退回页首导航。"""
    from omni_search.research import extract_relevant_quotes

    body = (
        "首页 导航 登录 注册 帮助中心\n\n"
        "本文介绍了大模型智能体如何利用长期记忆与混合检索架构提升复杂任务推理准确率。\n\n"
        "今天天气晴朗适合郊游散步。"
    )
    quotes = extract_relevant_quotes(body, "大模型智能体记忆系统混合检索", max_quotes=1)
    assert len(quotes) == 1
    assert "长期记忆" in quotes[0] and "混合检索" in quotes[0]


def test_extract_followup_terms_filters_generic_zh_academic_words():
    """面向/体系/构建等泛学术词不得挤占真实专业术语（如知识图谱/重排序）。"""
    from omni_search.research import extract_followup_terms

    cands = [
        SearchItem(
            id="g1",
            domain="academic",
            source="openalex",
            title="面向多跳问答的体系构建与重排序方法",
            url="https://example.com/g1",
            snippet="面向复杂场景构建体系，采用重排序与知识图谱。",
        ),
        SearchItem(
            id="g2",
            domain="academic",
            source="crossref",
            title="面向检索增强的体系构建和重排序评估",
            url="https://example.com/g2",
            snippet="面向工业场景构建标准体系并引入重排序与知识图谱。",
        ),
    ]
    terms = extract_followup_terms(cands, brief="大模型混合检索", max_terms=2)
    assert all(bad not in terms for bad in ("面向", "体系", "构建", "场景", "评估"))
    assert "知识图谱" in terms and "重排序" in terms


def test_extract_followup_terms_filters_pronouns_and_hyphen_stopwords():
    """英文自反提词应过滤 your / open-source 等代词与连字符泛词，保留真正技术实体。"""
    from omni_search.research import extract_followup_terms

    cands = [
        SearchItem(
            id="e1",
            domain="web",
            source="searxng",
            title="Build your open-source search with model2vec and fastembed",
            url="https://example.com/e1",
            snippet="Upgrade your open-source workflow using model2vec and fastembed.",
        ),
        SearchItem(
            id="e2",
            domain="code",
            source="github_direct",
            title="Your open-source model2vec + fastembed starter",
            url="https://github.com/a/b",
            snippet="Deploy your open-source pipeline with model2vec and fastembed.",
        ),
    ]
    terms = extract_followup_terms(cands, brief="local multimodal search", max_terms=2)
    assert "your" not in terms
    assert "open-source" not in terms
    assert set(terms) == {"model2vec", "fastembed"}


def test_select_diverse_verify_targets_demotes_google_news_redirect():
    """Google News RSS 加密跳转链应降权至直连原文之后，不浪费 verify_top_k 配额。"""
    from omni_search.research import _select_diverse_verify_targets

    gnews = SearchItem(
        id="gn1",
        domain="news",
        source="google_news_rss",
        title="Breaking AI Release",
        url="https://news.google.com/rss/articles/CBMiWWh0dHBzOi8vZXhhbXBsZS5jb20vYXJ0aWNsZQ?oc=5",
        score=0.95,
        extra={"evidence_tier": "T3"},
    )
    direct = SearchItem(
        id="d1",
        domain="news",
        source="hackernews",
        title="Direct Blog Post",
        url="https://blog.example.org/ai-release",
        score=0.40,
        extra={"evidence_tier": "T3"},
    )
    t1_doc = SearchItem(
        id="t1",
        domain="academic",
        source="arxiv",
        title="Official Paper",
        url="https://arxiv.org/abs/2601.00001",
        score=0.30,
        extra={"evidence_tier": "T1"},
    )
    picked = _select_diverse_verify_targets([gnews, direct, t1_doc], top_k=2)
    assert [c.url for c in picked] == [t1_doc.url, direct.url]

    # 若配额足够仍可保留作兜底，不会丢失候选
    picked_all = _select_diverse_verify_targets([gnews, direct, t1_doc], top_k=3)
    assert picked_all[-1].url == gnews.url
