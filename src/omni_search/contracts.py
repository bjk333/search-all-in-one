"""EngineContract v2、Coverage 派生与安全输出净化。"""

from __future__ import annotations

from enum import Enum
import os
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    OK = "ok"
    ZERO_HIT = "zero-hit"
    MISSING_CREDENTIAL = "missing-credential"
    TOOL_MISSING = "tool-missing"
    NETWORK_UNAVAILABLE = "network-unavailable"
    RATE_LIMITED = "rate-limited"
    QUOTA_EXHAUSTED = "quota-exhausted"
    FILTERED = "filtered"
    FORBIDDEN = "forbidden"
    DEADLINE = "deadline"
    BUDGET_TRUNCATED = "budget-truncated"
    USER_DISABLED = "user-disabled"
    NOT_RUN = "not-run"
    ERROR = "error"


_BLOCKED_CODES = {
    ReasonCode.MISSING_CREDENTIAL,
    ReasonCode.QUOTA_EXHAUSTED,
    ReasonCode.TOOL_MISSING,
    ReasonCode.NETWORK_UNAVAILABLE,
    ReasonCode.FORBIDDEN,
    ReasonCode.NOT_RUN,
}

# 零宽字符、BOM、双向控制字符与 Unicode Tag 块
_CTRL_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\U000e0000-\U000e007f]"
)

_SECRET_PARAM_RE = re.compile(
    r"^(api[_-]?key|apikey|token|access[_-]?token|auth|key|secret|password|passwd|sig|signature|client[_-]?secret)$",
    re.IGNORECASE,
)


def sanitize_text(text: str | None) -> str:
    """移除零宽/双向控制字符，但原样保留普通文本作为数据（不删除、不执行）。"""
    if not text:
        return ""
    return _CTRL_RE.sub("", str(text))


def redact_url_secrets(url: str | None) -> str:
    """对 URL 查询串中的凭据类参数值进行脱敏。"""
    if not url:
        return ""
    raw = sanitize_text(url).strip()
    try:
        parts = urlsplit(raw)
        if not parts.query:
            return raw
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        redacted = []
        changed = False
        for k, v in pairs:
            if _SECRET_PARAM_RE.match(k):
                redacted.append((k, "<redacted>"))
                changed = True
            else:
                redacted.append((k, v))
        if not changed:
            return raw
        new_query = urlencode(redacted, safe="<>")
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    except Exception:
        return raw


class RequestStats(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0


class EngineFailure(BaseModel):
    reason_code: ReasonCode
    reason: str


class SearchItem(BaseModel):
    id: str
    domain: str
    source: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    modality: str = "text"
    popularity: float | int | None = None
    unit: str = "none"
    score: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class EngineEnvelope(BaseModel):
    contract: str = "engine/2"
    engine: str
    domain: str
    results: list[SearchItem] = Field(default_factory=list)
    partial: bool = False
    elapsed_ms: int = 0
    requests: RequestStats = Field(default_factory=RequestStats)
    failure: EngineFailure | None = None
    warnings: list[str] = Field(default_factory=list)


class CoverageReport(BaseModel):
    requested: list[str] = Field(default_factory=list)
    executed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    reason_codes: dict[str, str] = Field(default_factory=dict)
    verdict: str = "no-engines"


class CompletenessReport(BaseModel):
    returned_count: int = 0
    total_candidates: int = 0
    truncated: bool = False
    per_source_limit: int = 0


class LedgerSummary(BaseModel):
    run_id: str
    elapsed_ms: int = 0
    budget_ms: int = 0
    budget_truncated: bool = False
    requests_attempted: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0


class SearchResponse(BaseModel):
    version: str
    query: str
    domains: list[str]
    results: list[SearchItem]
    coverage: CoverageReport
    completeness: CompletenessReport
    warnings: list[str] = Field(default_factory=list)
    ledger: LedgerSummary
    cached: bool = False
    low_confidence_negative: bool = False


def build_coverage(
    envelopes: list[EngineEnvelope],
    disabled_engines: list[str] | None = None,
) -> CoverageReport:
    """由池侧统一机械派生 Coverage 与 Verdict，引擎不自报状态。"""
    requested: list[str] = []
    executed: list[str] = []
    blocked: list[str] = []
    degraded: list[str] = []
    reasons: dict[str, str] = {}
    reason_codes: dict[str, str] = {}

    for d in disabled_engines or []:
        reason_codes[d] = ReasonCode.USER_DISABLED.value
        reasons[d] = "disabled by OMNI_DISABLE"

    for env in envelopes:
        requested.append(env.engine)
        has_results = len(env.results) > 0
        if env.failure is None:
            executed.append(env.engine)
            if not has_results:
                reason_codes[env.engine] = ReasonCode.ZERO_HIT.value
                reasons[env.engine] = "executed with 0 hits"
        else:
            rc = env.failure.reason_code
            reason_codes[env.engine] = rc.value
            reasons[env.engine] = env.failure.reason
            if has_results or env.partial:
                executed.append(env.engine)
                degraded.append(env.engine)
            elif rc in _BLOCKED_CODES:
                blocked.append(env.engine)
            else:
                degraded.append(env.engine)

    if not requested:
        verdict = "no-engines"
    elif not executed:
        verdict = "none"
    elif degraded:
        verdict = "degraded"
    elif blocked:
        verdict = "partial"
    else:
        verdict = "full"

    return CoverageReport(
        requested=requested,
        executed=executed,
        blocked=blocked,
        degraded=degraded,
        reasons=reasons,
        reason_codes=reason_codes,
        verdict=verdict,
    )


def compact_search_response(resp: SearchResponse, max_snippet_chars: int = 160) -> dict[str, Any]:
    """省 Token 精简视图：保留 100% 覆盖率与审计字段，压缩单条结果与冗余 extra。"""
    items: list[dict[str, Any]] = []
    for r in resp.results:
        snip = r.snippet.strip()
        if len(snip) > max_snippet_chars:
            snip = snip[:max_snippet_chars].rstrip() + "..."
        row: dict[str, Any] = {
            "title": r.title,
            "url": r.url,
            "domain": r.domain,
            "source": r.source,
            "snippet": snip,
        }
        if r.published_at:
            row["published_at"] = r.published_at
        for k in ("matched_line", "matched_page", "matched_locator", "file", "line", "rerank_score"):
            if r.extra.get(k) is not None:
                row[k] = r.extra[k]
        items.append(row)
    return {
        "version": resp.version,
        "query": resp.query,
        "domains": resp.domains,
        "results": items,
        "coverage": resp.coverage.model_dump(),
        "completeness": resp.completeness.model_dump(),
        "warnings": resp.warnings,
        "ledger": resp.ledger.model_dump(),
        "cached": resp.cached,
        "low_confidence_negative": resp.low_confidence_negative,
        "concise": True,
    }


_GH_CLI_TOKEN_CACHE: str = ""
_GH_CLI_TOKEN_PROBED: bool = False


def resolve_github_token(
    env: dict[str, str] | None = None,
    *,
    allow_gh_cli: bool = True,
) -> str:
    """只读解析 GitHub 凭据：优先 GITHUB_TOKEN -> GH_TOKEN -> 可选复用本机已登录的 `gh auth token`（进程内缓存，绝不落盘）。"""
    global _GH_CLI_TOKEN_CACHE, _GH_CLI_TOKEN_PROBED
    src = os.environ if env is None else env
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        val = (src.get(key) or "").strip()
        if val:
            return val
    if not allow_gh_cli:
        return ""
    if (src.get("OMNI_NO_GH_CLI_TOKEN") or "").strip().lower() in ("1", "true", "yes"):
        return ""
    if _GH_CLI_TOKEN_PROBED:
        return _GH_CLI_TOKEN_CACHE
    _GH_CLI_TOKEN_PROBED = True
    import shutil
    import subprocess

    gh_bin = shutil.which("gh")
    if not gh_bin:
        return ""
    try:
        proc = subprocess.run(
            [gh_bin, "auth", "token"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode == 0:
            tok = (proc.stdout or "").strip()
            if tok and "\n" not in tok:
                _GH_CLI_TOKEN_CACHE = tok
    except Exception:
        pass
    return _GH_CLI_TOKEN_CACHE
