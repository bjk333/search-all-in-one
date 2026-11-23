"""调研证据权威度分级、置信度评定、引文抽取、时效冲突审计与主张级引用对齐（从 research.py 解耦）。"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from omni_search.contracts import SearchItem
from omni_search.fetcher import FetchResult
from omni_search.research_models import (
    CitationAuditSummary,
    FreshnessConflictAudit,
    VerifiedEvidence,
)
from omni_search.research_planner import _STOPWORDS, _ZH_EDGE_STOP

_T1_HOSTS = (
    "arxiv.org",
    "doi.org",
    "ietf.org",
    "rfc-editor.org",
    "w3.org",
    "github.com",
    "gitlab.com",
    "codeberg.org",
    "gitee.com",
    "huggingface.co",
    "pypi.org",
    "npmjs.com",
    "crates.io",
    "developer.mozilla.org",
    "sourcegraph.com",
    "search.maven.org",
    "nuget.org",
    "rubygems.org",
    "packagist.org",
    "registry.modelcontextprotocol.io",
    "datatracker.ietf.org",
    "nvd.nist.gov",
    "osv.dev",
    "zenodo.org",
    "europepmc.org",
    "anaconda.org",
    "registry.terraform.io",
    "pkg.go.dev",
)
_T2_HOSTS = (
    "wikipedia.org",
    "wikidata.org",
    "wikimedia.org",
    "hub.docker.com",
    "openalex.org",
    "crossref.org",
    "openlibrary.org",
    "archive.org",
    "openstreetmap.org",
)
_T1_SOURCES = {
    "arxiv", "hf_papers", "github_direct", "github_issues", "gitlab", "codeberg", "gitee",
    "grep_app", "sourcegraph", "github_code", "local_code", "local_index",
    "mcps_repo", "mcps_packages", "huggingface_models", "hf_datasets", "mdn",
    "ietf_rfc", "nvd_cve", "osv_vulns", "zenodo", "europe_pmc",
    "npm_direct", "crates_direct", "pypi_ecosystems", "golang_ecosystems",
    "maven", "nuget", "rubygems", "packagist", "mcp_registry", "anaconda", "terraform_registry",
}
_T2_SOURCES = {
    "wikipedia", "wikipedia_zh", "wikidata", "wikimedia_commons", "dockerhub",
    "openalex", "crossref", "openlibrary", "archive_org", "osm_nominatim",
}

_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|doi\.org/10\.48550/arxiv\.)(\d{2})(\d{2})\.\d+", re.IGNORECASE)
_SNIPPET_DATE_RE = re.compile(
    r"^\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(19[89]\d|20[0-3]\d)|(19[89]\d|20[0-3]\d)[-/.]\d{1,2}[-/.]\d{1,2})",
    re.IGNORECASE,
)
_NAV_NOISE_RE = re.compile(
    r"you must be signed in|repository navigation|notifications|cookie|sign up|log in|privacy policy|edit links|jump to content",
    re.IGNORECASE,
)


def classify_evidence_tier(url: str, source: str = "") -> str:
    """确定性来源权威度分级：T1(一手规范/论文/代码仓) > T2(百科/图谱/索引) > T3(社区/媒体/普通网页)。"""
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if any(host == h or host.endswith("." + h) for h in _T1_HOSTS):
        return "T1"
    if host.startswith(("docs.", "developer.", "spec.", "standards.")) or host.endswith((".gov", ".edu")):
        return "T1"
    if source in _T1_SOURCES:
        return "T1"
    if any(host == h or host.endswith("." + h) for h in _T2_HOSTS) or source in _T2_SOURCES:
        return "T2"
    return "T3"


def compute_confidence(tier: str, matched_lanes: int, matched_sources: int, verified: bool) -> str:
    """透明可解释的证据置信度评定。"""
    score = 0
    if tier == "T1":
        score += 2
    elif tier == "T2":
        score += 1
    if verified:
        score += 2
    if matched_lanes >= 2 or matched_sources >= 2:
        score += 1
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def extract_pub_year(published_at: str | None, url: str = "", snippet: str = "") -> int | None:
    """从发布时间字段、arXiv/DOI URL 或摘要前导日期中提取确定性年份，绝不臆造。"""
    if published_at:
        m = _YEAR_RE.search(str(published_at))
        if m:
            return int(m.group(1))
    if url:
        am = _ARXIV_ID_RE.search(url)
        if am:
            yy = int(am.group(1))
            return 2000 + yy if yy <= 39 else 1900 + yy
    if snippet:
        sm = _SNIPPET_DATE_RE.search(snippet[:60])
        if sm:
            yr_str = sm.group(1) or sm.group(2)
            if yr_str:
                return int(yr_str)
    return None


_ASCII_GLUED_TOKEN_RE = re.compile(r"^[A-Za-z0-9_#/-]{60,}$")


def extract_relevant_quotes(text: str, query: str, max_quotes: int = 2) -> list[str]:
    """借鉴 Quote Grounding：过滤导航/登录噪音句，按查询词命中密度抽取高信噪比证据句。"""
    if not text:
        return []

    qterms_raw = [
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", query)
        if t.lower() not in _STOPWORDS
    ]
    qterms: list[str] = []
    for t in qterms_raw:
        if re.fullmatch(r"[\u4e00-\u9fff]+", t) and len(t) > 4:
            for i in range(len(t) - 1):
                g = t[i : i + 2]
                if g[0] not in _ZH_EDGE_STOP and g[-1] not in _ZH_EDGE_STOP:
                    qterms.append(g)
        else:
            qterms.append(t)
    qterms = list(dict.fromkeys(qterms))
    sents = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n{2,}", text)
    scored: list[tuple[float, str]] = []
    for raw_s in sents:
        s = re.sub(r"\s+", " ", raw_s).strip()
        if len(s) < 25 or _NAV_NOISE_RE.search(s):
            continue
        # 仅跳过纯 ASCII 无空格超长粘连标签串（避免误杀无空格中文长句，且排除普通 URL）
        if any(
            _ASCII_GLUED_TOKEN_RE.fullmatch(tok) and not tok.lower().startswith(("http://", "https://"))
            for tok in s.split()
        ):
            continue
        low = s.lower()
        hits = sum(1 for t in qterms if t in low) if qterms else 0
        # 偏好命中词多且信息长度适中的句子
        score = hits * 2.0 + min(len(s), 260) / 260.0
        if hits > 0 or not qterms:
            scored.append((score, s[:280]))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    for _, s in scored:
        if s not in out:
            out.append(s)
        if len(out) >= max_quotes:
            break
    return out


def audit_freshness_and_conflicts(
    candidates: list[SearchItem],
    current_year: int | None = None,
) -> FreshnessConflictAudit:
    """标注每条候选时效等级，并检测跨源日期冲突与 Top 候选整体陈旧风险。"""
    now_yr = current_year or datetime.now(timezone.utc).year
    audit = FreshnessConflictAudit(current_year=now_yr)
    title_years: dict[str, list[tuple[str, int]]] = {}

    for c in candidates:
        yr = extract_pub_year(c.published_at, c.url, c.snippet)
        src_years: dict[str, int] = dict(c.extra.get("source_years") or {})
        if yr is not None:
            src_years.setdefault(c.source, yr)
        elif src_years:
            yr = max(src_years.values())

        c.extra["pub_year"] = yr
        if src_years:
            c.extra["source_years"] = src_years

        if yr is None:
            c.extra["freshness"] = "unknown"
            audit.unknown_count += 1
        elif now_yr - yr <= 1:
            c.extra["freshness"] = "recent"
            audit.recent_count += 1
        elif now_yr - yr <= 4:
            c.extra["freshness"] = "aging"
            audit.aging_count += 1
        else:
            c.extra["freshness"] = "stale"
            audit.stale_count += 1

        # 1. 单条跨源年份差异检测（>= 2 年）
        if len(src_years) >= 2:
            lo, hi = min(src_years.values()), max(src_years.values())
            if hi - lo >= 2:
                detail = ", ".join(f"{s}={y}" for s, y in sorted(src_years.items()))
                audit.date_discrepancies.append(f"跨源日期冲突 [{c.title[:50]}]: {detail}")

        # 2. 同标题跨条目年份差异检测
        norm_title = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", c.title.lower()).strip()
        if norm_title and len(norm_title) >= 8 and yr is not None:
            title_years.setdefault(norm_title, []).append((c.source, yr))

    for tkey, pairs in title_years.items():
        yrs = [y for _, y in pairs]
        if len(set(yrs)) >= 2 and (max(yrs) - min(yrs)) >= 2:
            detail = ", ".join(f"{s}={y}" for s, y in pairs)
            msg = f"同标题不同年份版本 [{tkey[:45]}]: {detail}"
            if msg not in audit.date_discrepancies:
                audit.date_discrepancies.append(msg)

    top_slice = candidates[:5]
    if top_slice:
        top_stale = sum(1 for c in top_slice if c.extra.get("freshness") == "stale")
        top_recent = sum(1 for c in top_slice if c.extra.get("freshness") == "recent")
        if top_stale * 2 >= len(top_slice) and top_stale > 0:
            audit.freshness_warnings.append(
                f"Top-{len(top_slice)} 高分候选中陈旧条目占比偏高（{top_stale}/{len(top_slice)} 条 <= {now_yr - 5} 年），请结合近期资料复核现状"
            )
        elif top_recent == 0 and (audit.aging_count + audit.stale_count) > 0:
            audit.freshness_warnings.append(
                f"Top-{len(top_slice)} 候选缺乏近 2 年（{now_yr - 1}–{now_yr}）新证据，结论可能滞后"
            )

    return audit


def audit_verified_citations(brief: str, verified_evidence: list[VerifiedEvidence]) -> CitationAuditSummary | None:
    """复用 verifier 对齐引擎，零额外网络请求核验已抓取的 Top-K 证据正文是否真正支撑研究主题。"""
    if not verified_evidence:
        return None
    from omni_search.verifier import _evaluate_claim_against_page

    counts = {"supported": 0, "partial": 0, "unsupported": 0, "unreachable": 0}
    flagged: list[str] = []
    for ev in verified_evidence:
        if not ev.verified:
            counts["unreachable"] += 1
            flagged.append(ev.url)
            continue
        body_parts = [ev.excerpt or ""]
        if ev.relevant_quotes:
            body_parts.extend(ev.relevant_quotes)
        page = FetchResult(
            ok=True,
            url=ev.url,
            status_code=ev.status_code,
            title=ev.title,
            content="\n".join(p for p in body_parts if p),
            untrusted_content=True,
        )
        item = _evaluate_claim_against_page(brief, ev.url, page)
        v = item.verdict if item.verdict in counts else "unsupported"
        counts[v] += 1
        if v == "unsupported":
            flagged.append(ev.url)

    total = len(verified_evidence)
    return CitationAuditSummary(
        total_checked=total,
        supported_count=counts["supported"],
        partial_count=counts["partial"],
        unsupported_count=counts["unsupported"],
        unreachable_count=counts["unreachable"],
        grounding_rate=round(counts["supported"] / total, 4) if total else 0.0,
        flagged_urls=flagged,
    )


__all__ = [
    "_ARXIV_ID_RE",
    "_NAV_NOISE_RE",
    "_SNIPPET_DATE_RE",
    "_T1_HOSTS",
    "_T1_SOURCES",
    "_T2_HOSTS",
    "_T2_SOURCES",
    "_YEAR_RE",
    "audit_freshness_and_conflicts",
    "audit_verified_citations",
    "classify_evidence_tier",
    "compute_confidence",
    "extract_pub_year",
    "extract_relevant_quotes",
]
