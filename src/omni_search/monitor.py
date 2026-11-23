"""monitor: 基于确定性调研快照的主题持续增量追踪（仅输出新增 T1 证据 / 新 Release / 新 CVE 变更）。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pathlib
import re
from typing import Any

from pydantic import BaseModel, Field

from omni_search.pool import FederatedPool
from omni_search.research import (
    DiffItem,
    FetcherFn,
    ResearchReport,
    diff_research_reports,
    run_research_rounds,
)


class MonitorResult(BaseModel):
    topic: str
    mode: str
    snapshot_path: str
    checked_at: str
    is_baseline: bool
    has_changes: bool
    retained_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    added_t1: list[DiffItem] = Field(default_factory=list)
    added_items: list[DiffItem] = Field(default_factory=list)
    removed_items: list[DiffItem] = Field(default_factory=list)
    newly_failed_verify: list[str] = Field(default_factory=list)
    tier_delta: dict[str, int] = Field(default_factory=dict)
    total_candidates: int = 0
    gaps: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


def _snapshot_file(state_dir: pathlib.Path, topic: str, mode: str) -> pathlib.Path:
    norm = f"{mode.strip().lower()}|{' '.join(topic.strip().lower().split())}"
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", topic.strip().lower()).strip("-")[:32] or "topic"
    return state_dir / f"{slug}_{digest}.json"


def render_monitor_markdown(res: MonitorResult) -> str:
    lines = [
        f"# 主题增量监控报告：{res.topic}",
        "",
        f"- **检测时间**：`{res.checked_at}` · **模式**：`{res.mode}` · **耗时**：`{res.elapsed_ms} ms`",
        f"- **快照路径**：`{res.snapshot_path}`",
    ]
    if res.is_baseline:
        lines.append(f"- **状态**：首次建立基线（共收录 `{res.total_candidates}` 条候选）")
    elif not res.has_changes:
        lines.append(f"- **状态**：无新增或移除变更（保留 `{res.retained_count}` 条）")
    else:
        lines.append(
            f"- **状态**：检测到变更（新增 `{res.added_count}` / 移除 `{res.removed_count}` / 保留 `{res.retained_count}`，其中新增 T1 `{len(res.added_t1)}` 条）"
        )
    if res.added_t1:
        lines.extend(["", "## 新增 T1 高权威证据", ""])
        for it in res.added_t1[:15]:
            lines.append(f"- **[{it.domain}/{it.source}]** [{it.title}]({it.url})")
    other_added = [it for it in res.added_items if it.evidence_tier != "T1"]
    if other_added:
        lines.extend(["", "## 其他新增条目（T2/T3）", ""])
        for it in other_added[:15]:
            lines.append(f"- `[{it.evidence_tier}]` [{it.title}]({it.url}) ({it.source})")
    if res.newly_failed_verify:
        lines.extend(["", "## 原已核验但本次失效链接", ""])
        for u in res.newly_failed_verify:
            lines.append(f"- `{u}`")
    if res.gaps:
        lines.extend(["", "## 覆盖缺口", ""])
        for g in res.gaps[:12]:
            lines.append(f"- `{g}`")
    lines.append("")
    return "\n".join(lines)


async def run_monitor_check(
    pool: FederatedPool,
    topic: str,
    state_dir: pathlib.Path | str,
    mode: str = "standard",
    domains: list[str] | None = None,
    limit: int = 4,
    verify_top_k: int = 2,
    budget_ms: int = 25000,
    fetcher: FetcherFn | None = None,
) -> MonitorResult:
    """运行一轮主题监控：与历史快照 diff 后刷新快照文件并返回结构化增量变更。"""
    sdir = pathlib.Path(state_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    snap_path = _snapshot_file(sdir, topic, mode)

    prev_report: ResearchReport | None = None
    if snap_path.exists():
        try:
            prev_report = ResearchReport.model_validate(json.loads(snap_path.read_text(encoding="utf-8")))
        except Exception:
            prev_report = None

    curr_report = await run_research_rounds(
        pool=pool,
        brief=topic,
        domains=domains,
        mode=mode,
        limit=limit,
        verify_top_k=verify_top_k,
        budget_ms=budget_ms,
        render_markdown=False,
        fetcher=fetcher,
    )

    snap_path.write_text(
        json.dumps(curr_report.model_dump(exclude={"markdown", "bibtex"}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    if prev_report is None:
        return MonitorResult(
            topic=topic,
            mode=curr_report.mode,
            snapshot_path=str(snap_path),
            checked_at=now_iso,
            is_baseline=True,
            has_changes=False,
            retained_count=len(curr_report.candidates),
            total_candidates=len(curr_report.candidates),
            gaps=curr_report.gaps,
            elapsed_ms=curr_report.elapsed_ms,
        )

    diff = diff_research_reports(prev_report, curr_report)
    added_t1 = [it for it in diff.added if it.evidence_tier == "T1"]
    has_changes = bool(diff.added or diff.removed or diff.newly_failed_verify)
    return MonitorResult(
        topic=topic,
        mode=curr_report.mode,
        snapshot_path=str(snap_path),
        checked_at=now_iso,
        is_baseline=False,
        has_changes=has_changes,
        retained_count=diff.retained_count,
        added_count=len(diff.added),
        removed_count=len(diff.removed),
        added_t1=added_t1,
        added_items=diff.added,
        removed_items=diff.removed,
        newly_failed_verify=diff.newly_failed_verify,
        tier_delta=diff.tier_delta,
        total_candidates=len(curr_report.candidates),
        gaps=curr_report.gaps,
        elapsed_ms=curr_report.elapsed_ms,
    )
