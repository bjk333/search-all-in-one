"""多 Agent 并行调研工单导出与严审合单模块（从 research.py 解耦）。"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import TYPE_CHECKING

from omni_search import __version__
from omni_search.contracts import SearchResponse
from omni_search.fetcher import safe_fetch_url
from omni_search.pool import normalize_url_key
from omni_search.research_audit import audit_freshness_and_conflicts, compute_confidence
from omni_search.research_evidence import _merge_lane_candidates, _verify_candidates_with_fallback
from omni_search.research_models import (
    FetcherFn,
    LaneExecution,
    MergeAudit,
    ResearchReport,
    _to_lane_exec,
)
from omni_search.research_planner import ResearchPlan, plan_research_lanes
from omni_search.research_render import render_citations_bibtex, render_research_markdown


def export_research_workorders(
    out_dir: pathlib.Path | str,
    brief: str,
    domains: list[str] | None = None,
    mode: str = "standard",
    extra_terms: list[str] | None = None,
    include_local: bool = False,
) -> ResearchPlan:
    """导出多 Agent 并行扇出工单（plan.json + agents/<lane>.md），不联网。"""
    plan = plan_research_lanes(brief, domains=domains, mode=mode, extra_terms=extra_terms, include_local=include_local)
    base = pathlib.Path(out_dir)
    agents_dir = base / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (base / "plan.json").write_text(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for ln in plan.lanes:
        dom_arg = ",".join(ln.domains)
        md = "\n".join(
            [
                f"# Lane 工单：`{ln.name}`",
                "",
                f"- **Plan ID**：`{plan.plan_id}`（合单强制核验，不可篡改）",
                f"- **调研主题**：{plan.brief}",
                f"- **视角目标**：{ln.purpose or ln.name}",
                f"- **目标域**：`{dom_arg}`",
                f"- **查询词**：`{ln.query}`",
                "",
                "## 执行命令",
                "",
                "```bash",
                f'uv --directory "D:/01 项目/omni-search" run omni search "{ln.query}" -d {dom_arg} --limit 5 --out "{ln.name}.raw.json"',
                "```",
                "",
                "## 返回契约",
                "",
                f"将结果封装为 `agents/{ln.name}.result.json`：",
                "",
                "```json",
                json.dumps(
                    {
                        "plan_id": plan.plan_id,
                        "lane": ln.name,
                        "query": ln.query,
                        "domains": ln.domains,
                        "response": "<上述 omni search 的完整 JSON 对象>",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
        (agents_dir / f"{ln.name}.md").write_text(md, encoding="utf-8")
    return plan


def _response_fingerprint(resp: SearchResponse) -> str:
    sig = {
        "verdict": resp.coverage.verdict,
        "urls": sorted(normalize_url_key(r.url) or r.title.lower() for r in resp.results),
    }
    return hashlib.sha256(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


async def merge_lane_results(
    work_dir: pathlib.Path | str,
    verify_top_k: int = 3,
    budget_ms: int = 25000,
    render_markdown: bool = False,
    render_bibtex: bool = False,
    fetcher: FetcherFn | None = None,
) -> ResearchReport:
    """严格按 plan.json 合并多 Agent 的 *.result.json（拒收不符、去重同副本、冲突全排除）。"""
    t0 = time.perf_counter()
    base = pathlib.Path(work_dir)
    plan_file = base / "plan.json"
    if not plan_file.exists():
        raise FileNotFoundError(f"plan.json not found in {base}")
    plan = ResearchPlan.model_validate(json.loads(plan_file.read_text(encoding="utf-8")))
    expected_lanes = {ln.name: ln for ln in plan.lanes}

    search_roots = [base / "agents", base]
    result_files: list[pathlib.Path] = []
    seen_paths: set[pathlib.Path] = set()
    for root in search_roots:
        if root.exists():
            for fp in sorted(root.glob("*.result.json")):
                res_p = fp.resolve()
                if res_p not in seen_paths:
                    seen_paths.add(res_p)
                    result_files.append(fp)

    audit = MergeAudit(plan_id=plan.plan_id)
    gaps: list[str] = []
    by_lane: dict[str, list[tuple[str, str, SearchResponse]]] = {}

    for fp in result_files:
        rel = fp.name
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            msg = f"{rel}: invalid json ({exc})"
            audit.rejected.append(msg)
            gaps.append(f"merge rejected {msg}")
            continue

        pid = raw.get("plan_id")
        lane_name = raw.get("lane")
        q = raw.get("query")
        if pid != plan.plan_id:
            msg = f"{rel}: plan_id mismatch ({pid} != {plan.plan_id})"
            audit.rejected.append(msg)
            gaps.append(f"merge rejected {msg}")
            continue
        if lane_name not in expected_lanes:
            msg = f"{rel}: unexpected lane '{lane_name}'"
            audit.rejected.append(msg)
            gaps.append(f"merge rejected {msg}")
            continue
        if q is not None and str(q).strip() != expected_lanes[lane_name].query:
            msg = f"{rel}: query mismatch for lane '{lane_name}'"
            audit.rejected.append(msg)
            gaps.append(f"merge rejected {msg}")
            continue
        try:
            resp = SearchResponse.model_validate(raw["response"])
        except Exception as exc:
            msg = f"{rel}: invalid SearchResponse ({exc})"
            audit.rejected.append(msg)
            gaps.append(f"merge rejected {msg}")
            continue

        fp_hash = _response_fingerprint(resp)
        by_lane.setdefault(lane_name, []).append((rel, fp_hash, resp))
        audit.source_files.setdefault(lane_name, []).append(rel)

    lane_pairs: list[tuple[str, SearchResponse]] = []
    lane_execs: list[LaneExecution] = []
    for ln in plan.lanes:
        entries = by_lane.get(ln.name, [])
        if not entries:
            audit.missing_lanes.append(ln.name)
            gaps.append(f"merge missing lane={ln.name}")
            continue
        unique_hashes = {h for _, h, _ in entries}
        if len(unique_hashes) > 1:
            audit.conflicts.append(ln.name)
            files_list = ", ".join(f for f, _, _ in entries)
            gaps.append(f"merge conflict excluded lane={ln.name} files=[{files_list}]")
            continue
        if len(entries) > 1:
            for dup_file, _, _ in entries[1:]:
                audit.deduplicated.append(f"{ln.name}:{dup_file}")
        _, _, chosen = entries[0]
        audit.accepted_lanes.append(ln.name)
        lane_pairs.append((ln.name, chosen))
        lane_execs.append(_to_lane_exec(ln.name, chosen))
        for eng, rc in chosen.coverage.reason_codes.items():
            if rc not in ("zero-hit", "user-disabled"):
                gaps.append(f"lane={ln.name} engine={eng} status={rc}")

    candidates = _merge_lane_candidates(lane_pairs, brief=plan.brief)

    # 可选 Top-K 原文安全核验（支持失败顺延补位 + Quote Grounding）
    fetch_fn = fetcher or safe_fetch_url
    deadline = t0 + max(1.0, budget_ms / 1000.0)
    verified_evidence, verified_urls, ver_gaps, _ = await _verify_candidates_with_fallback(
        candidates=candidates,
        brief=plan.brief,
        verify_top_k=verify_top_k,
        deadline=deadline,
        fetch_fn=fetch_fn,
    )
    gaps.extend(ver_gaps)

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

    report = ResearchReport(
        version=__version__,
        plan_id=plan.plan_id,
        brief=plan.brief,
        mode=plan.mode,
        extra_terms=plan.extra_terms,
        lanes=lane_execs,
        candidates=candidates,
        verified_evidence=verified_evidence,
        tier_counts=tier_counts,
        freshness_audit=freshness_audit,
        merge_audit=audit,
        gaps=list(dict.fromkeys(gaps)),
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        budget_ms=budget_ms,
        budget_truncated=False,
    )
    if render_markdown:
        report.markdown = render_research_markdown(report)
    if render_bibtex:
        report.bibtex = render_citations_bibtex(report)
    return report


__all__ = [
    "_response_fingerprint",
    "export_research_workorders",
    "merge_lane_results",
]
