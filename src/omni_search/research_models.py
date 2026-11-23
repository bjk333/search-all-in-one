"""research_models: 深度调研核心数据契约模型（零反向依赖，供 research / audit / render / workorders 共享）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from omni_search.contracts import SearchItem, SearchResponse
from omni_search.fetcher import FetchResult

FetcherFn = Callable[..., Awaitable[FetchResult]]


class LaneExecution(BaseModel):
    name: str
    query: str
    domains: list[str]
    verdict: str
    result_count: int
    executed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    reason_codes: dict[str, str] = Field(default_factory=dict)
    elapsed_ms: int = 0


class VerifiedEvidence(BaseModel):
    url: str
    title: str
    evidence_tier: str = "T3"
    verified: bool
    status_code: int | None = None
    excerpt: str = ""
    relevant_quotes: list[str] = Field(default_factory=list)
    untrusted_content: bool = True
    reason_code: str | None = None
    reason: str | None = None


class FreshnessConflictAudit(BaseModel):
    current_year: int
    recent_count: int = 0
    aging_count: int = 0
    stale_count: int = 0
    unknown_count: int = 0
    date_discrepancies: list[str] = Field(default_factory=list)
    freshness_warnings: list[str] = Field(default_factory=list)


class MergeAudit(BaseModel):
    plan_id: str
    accepted_lanes: list[str] = Field(default_factory=list)
    missing_lanes: list[str] = Field(default_factory=list)
    deduplicated: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    source_files: dict[str, list[str]] = Field(default_factory=dict)


class DiffItem(BaseModel):
    title: str
    url: str
    domain: str
    source: str
    evidence_tier: str = "T3"


class ResearchDiff(BaseModel):
    prev_plan_id: str
    curr_plan_id: str
    prev_brief: str = ""
    retained_count: int = 0
    added: list[DiffItem] = Field(default_factory=list)
    removed: list[DiffItem] = Field(default_factory=list)
    newly_verified: list[str] = Field(default_factory=list)
    newly_failed_verify: list[str] = Field(default_factory=list)
    tier_delta: dict[str, int] = Field(default_factory=dict)


class CitationAuditSummary(BaseModel):
    total_checked: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    unreachable_count: int = 0
    grounding_rate: float = 0.0
    flagged_urls: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    version: str
    plan_id: str
    brief: str
    mode: str = "standard"
    extra_terms: list[str] = Field(default_factory=list)
    lanes: list[LaneExecution]
    gap_fills: list[LaneExecution] = Field(default_factory=list)
    candidates: list[SearchItem]
    verified_evidence: list[VerifiedEvidence] = Field(default_factory=list)
    tier_counts: dict[str, int] = Field(default_factory=dict)
    freshness_audit: FreshnessConflictAudit | None = None
    citation_audit: CitationAuditSummary | None = None
    merge_audit: MergeAudit | None = None
    diff: ResearchDiff | None = None
    gaps: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    budget_ms: int = 0
    budget_truncated: bool = False
    markdown: str | None = None
    bibtex: str | None = None


def _to_lane_exec(name: str, resp: SearchResponse) -> LaneExecution:
    return LaneExecution(
        name=name,
        query=resp.query,
        domains=resp.domains,
        verdict=resp.coverage.verdict,
        result_count=len(resp.results),
        executed=resp.coverage.executed,
        blocked=resp.coverage.blocked,
        degraded=resp.coverage.degraded,
        reason_codes=resp.coverage.reason_codes,
        elapsed_ms=resp.ledger.elapsed_ms,
    )
