"""omni_research v2: 多模式全网深度调研编排器（差异化 Lane -> 缺口补查 -> 多样性原文核验 -> 证据分级 -> Markdown 简报）。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Any

from omni_search import __version__
from omni_search.contracts import SearchItem, SearchResponse, sanitize_text  # noqa: F401
from omni_search.fetcher import FetchResult, safe_fetch_url
from omni_search.pool import FederatedPool, normalize_url_key
from omni_search.research_audit import (  # noqa: F401 - re-exported for backward compatibility
    _ARXIV_ID_RE,
    _NAV_NOISE_RE,
    _SNIPPET_DATE_RE,
    _T1_HOSTS,
    _T1_SOURCES,
    _T2_HOSTS,
    _T2_SOURCES,
    _YEAR_RE,
    audit_freshness_and_conflicts,
    audit_verified_citations,
    classify_evidence_tier,
    compute_confidence,
    extract_pub_year,
    extract_relevant_quotes,
)
from omni_search.research_evidence import (  # noqa: F401 - re-exported for backward compatibility
    _is_low_value_redirect_url,
    _merge_lane_candidates,
    _select_diverse_verify_targets,
    _verify_candidates_with_fallback,
)
from omni_search.research_models import (  # noqa: F401 - re-exported for backward compatibility
    CitationAuditSummary,
    DiffItem,
    FetcherFn,
    FreshnessConflictAudit,
    LaneExecution,
    MergeAudit,
    ResearchDiff,
    ResearchReport,
    VerifiedEvidence,
    _to_lane_exec,
)
from omni_search.research_planner import (  # noqa: F401 - re-exported for backward compatibility
    VALID_RESEARCH_MODES,
    ResearchLane,
    ResearchPlan,
    _DEEPEN_IGNORE,
    _GENERIC_BROAD_TERMS,
    _STOPWORDS,
    _ZH_EDGE_STOP,
    _ZH_SPLIT_RE,
    _compute_query_relevance,
    _extract_candidate_terms,
    _is_cjk_dominant,
    _token_specificity_score,
    extract_followup_terms,
    plan_research_lanes,
    widen_query,
)
from omni_search.research_render import (  # noqa: F401 - re-exported for backward compatibility
    _escape_bibtex,
    _make_citekey,
    compact_research_report,
    diff_research_reports,
    render_citations_bibtex,
    render_research_markdown,
)
from omni_search.research_workorders import (  # noqa: F401 - re-exported for backward compatibility
    _response_fingerprint,
    export_research_workorders,
    merge_lane_results,
)


async def run_research_rounds(
    pool: FederatedPool,
    brief: str,
    domains: list[str] | None = None,
    mode: str = "standard",
    extra_terms: list[str] | None = None,
    limit: int = 4,
    verify_top_k: int = 3,
    budget_ms: int = 35000,
    render_markdown: bool = False,
    render_bibtex: bool = False,
    include_local: bool = False,
    deepen: bool = False,
    fetcher: FetcherFn | None = None,
    progress_cb: Callable[[int, int, str], Any] | None = None,
) -> ResearchReport:
    """执行确定性多轮全网深度调研（不调用 LLM）：差异化发现 -> 缺口宽词/自反追问补查 -> 多样性 Top-K 原文核验。"""
    t0 = time.perf_counter()
    deadline = t0 + max(1.0, budget_ms / 1000.0)

    def _emit_progress(step: int, total: int, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(step, total, msg)
            except Exception:
                pass

    plan = plan_research_lanes(brief, domains=domains, mode=mode, extra_terms=extra_terms, include_local=include_local)
    _emit_progress(1, 4, f"Round 1: launching {len(plan.lanes)} lanes ({plan.mode})")
    if fetcher is not None:
        fetch_fn = fetcher
    elif pool.disk_cache is not None:
        async def _cached_fetch(url: str, max_chars: int = 3000, timeout_sec: float = 10.0) -> FetchResult:
            return await safe_fetch_url(
                url,
                max_chars=max_chars,
                timeout_sec=timeout_sec,
                disk_cache=pool.disk_cache,
            )

        fetch_fn = _cached_fetch
    else:
        fetch_fn = safe_fetch_url

    # === Round 1: 多 Lane 并发发现 ===
    r1_budget_ms = max(1000, int((deadline - time.perf_counter()) * 1000 * 0.6))
    r1_tasks = [
        pool.search(query=ln.query, domains=ln.domains, limit=limit, max_results=limit * 3, budget_ms=r1_budget_ms)
        for ln in plan.lanes
    ]
    r1_resps: list[SearchResponse] = list(await asyncio.gather(*r1_tasks))
    lane_pairs: list[tuple[str, SearchResponse]] = [
        (ln.name, resp) for ln, resp in zip(plan.lanes, r1_resps)
    ]
    lane_execs = [_to_lane_exec(ln_name, resp) for ln_name, resp in lane_pairs]

    # === Round 2: 0 命中缺口宽词补查 ===
    gap_fills: list[LaneExecution] = []
    gaps: list[str] = []
    budget_truncated = False
    wider = widen_query(plan.brief, max_terms=3)

    gap_tasks: list[tuple[ResearchLane, Awaitable[SearchResponse]]] = []
    for ln, resp in zip(plan.lanes, r1_resps):
        for eng, rc in resp.coverage.reason_codes.items():
            if rc not in ("zero-hit", "user-disabled"):
                gaps.append(f"lane={ln.name} engine={eng} status={rc}")
        if len(resp.results) == 0 and wider.lower() != ln.query.lower():
            rem_ms = int((deadline - time.perf_counter()) * 1000)
            if rem_ms > 1500:
                gap_tasks.append(
                    (
                        ln,
                        pool.search(
                            query=wider,
                            domains=ln.domains,
                            limit=limit,
                            max_results=limit * 2,
                            budget_ms=min(rem_ms - 500, 10000),
                        ),
                    )
                )
            else:
                budget_truncated = True
                gaps.append(f"lane={ln.name} 0-hit gap-fill skipped due to budget")

    if gap_tasks:
        gf_resps = list(await asyncio.gather(*(t for _, t in gap_tasks)))
        for (ln, _), resp in zip(gap_tasks, gf_resps):
            gf_name = f"{ln.name}:gap-fill"
            gap_fills.append(_to_lane_exec(gf_name, resp))
            lane_pairs.append((gf_name, resp))
            if len(resp.results) == 0:
                gaps.append(f"lane={ln.name} remained 0-hit after wider query '{wider}'")

    candidates = _merge_lane_candidates(lane_pairs, brief=plan.brief)
    _emit_progress(2, 4, f"Round 2 complete: {len(candidates)} merged candidates")

    # === Round 2b: 可选确定性自反追问（从首轮多候选交叉共振中提炼新术语定向深挖）===
    if deepen and candidates:
        rem_ms = int((deadline - time.perf_counter()) * 1000)
        followup_terms = extract_followup_terms(candidates, plan.brief, max_terms=2)
        if followup_terms:
            if rem_ms > 2000:
                pub_doms = list(dict.fromkeys(d for ln in plan.lanes for d in ln.domains if d != "local"))
                target_doms = pub_doms[:3] if pub_doms else ["web", "academic", "code"]
                deep_query = f"{wider} {' '.join(followup_terms)}".strip()
                deep_resp = await pool.search(
                    query=deep_query,
                    domains=target_doms,
                    limit=limit,
                    max_results=limit * 2,
                    budget_ms=min(rem_ms - 500, 10000),
                )
                gap_fills.append(_to_lane_exec("followup:deepen", deep_resp))
                lane_pairs.append(("followup:deepen", deep_resp))
                candidates = _merge_lane_candidates(lane_pairs, brief=plan.brief)
            else:
                budget_truncated = True
                gaps.append("followup:deepen skipped due to budget")

    # === Round 3: 主机名多样性 + 失败顺延补位原文核验 (omni_fetch + Quote Grounding) ===
    _emit_progress(3, 4, f"Round 3: verifying top-{verify_top_k} primary sources")
    verified_evidence, verified_urls, ver_gaps, ver_trunc = await _verify_candidates_with_fallback(
        candidates=candidates,
        brief=plan.brief,
        verify_top_k=verify_top_k,
        deadline=deadline,
        fetch_fn=fetch_fn,
    )
    gaps.extend(ver_gaps)
    budget_truncated = budget_truncated or ver_trunc

    # 根据原文核验结果刷新候选置信度并统计证据分级分布
    tier_counts = {"T1": 0, "T2": 0, "T3": 0}
    for c in candidates:
        tier = str(c.extra.get("evidence_tier") or "T3")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        is_ver = normalize_url_key(c.url) in verified_urls
        c.extra["verified"] = is_ver
        c.extra["confidence"] = compute_confidence(
            tier,
            len(c.extra.get("matched_lanes") or []),
            len(c.extra.get("matched_sources") or []),
            verified=is_ver,
        )

    freshness_audit = audit_freshness_and_conflicts(candidates)
    for fw in freshness_audit.freshness_warnings:
        gaps.append(f"freshness: {fw}")
    for dd in freshness_audit.date_discrepancies:
        gaps.append(f"conflict: {dd}")

    citation_audit = audit_verified_citations(plan.brief, verified_evidence)
    if citation_audit and citation_audit.unsupported_count > 0:
        for fu in citation_audit.flagged_urls:
            if any(ev.url == fu and ev.verified for ev in verified_evidence):
                gaps.append(f"citation-unsupported: {fu}")

    report = ResearchReport(
        version=__version__,
        plan_id=plan.plan_id,
        brief=plan.brief,
        mode=plan.mode,
        extra_terms=plan.extra_terms,
        lanes=lane_execs,
        gap_fills=gap_fills,
        candidates=candidates,
        verified_evidence=verified_evidence,
        tier_counts=tier_counts,
        freshness_audit=freshness_audit,
        citation_audit=citation_audit,
        gaps=list(dict.fromkeys(gaps)),
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        budget_ms=budget_ms,
        budget_truncated=budget_truncated,
    )
    if render_markdown:
        report.markdown = render_research_markdown(report)
    if render_bibtex:
        report.bibtex = render_citations_bibtex(report)
    _emit_progress(4, 4, f"Done: {len(candidates)} candidates, {sum(1 for v in verified_evidence if v.verified)} verified")
    return report
