"""verifier: 主张级引用核验引擎（Citation Grounding / Claim Verification）。

支持从 Markdown 报告自动抽取带 URL 的引用主张，或直接核验结构化 {claim, url} 列表：
- 并发安全拉取所有去重后的引用 URL（完整 SSRF 阻断 + 磁盘缓存 + untrusted_content 标记）；
- 确定性提取硬事实锚点（版本号、CVE、数字/单位、技术缩写）与语义关键词，在原文中做对齐核验；
- 产出审计级判定：supported / partial / unsupported / unreachable / blocked，定位潜在引用幻觉。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
import time
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from omni_search import __version__
from omni_search.fetcher import FetchResult, redact_url_secrets, safe_fetch_url


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"(?<!\()(https?://[^\s<>)\"']+)")
_HARD_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:CVE-\d{4}-\d{4,7}|v?\d+\.\d+(?:\.\d+)*|"
    r"(?i:\d+(?:\.\d+)?\s*(?:ms|us|ns|sec|kb|mb|gb|tb|kib|mib|gib|x))|"
    r"\d+(?:\.\d+)?\s*(?:倍|万|亿)|"
    r"\d+(?:\.\d+)?%|\d{1,3}(?:,\d{3})+|\d{4}|[A-Z][A-Z0-9_-]{2,})(?![A-Za-z0-9_])"
)
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{2,}")
_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_EDGE_STOP = frozenset("的了和与在中是对为及等将把被从到之其于或以所能可这那该")
_VERSION_TOK_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)*)$", re.IGNORECASE)
_UNIT_NUM_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(ms|us|ns|sec|kb|mb|gb|tb|kib|mib|gib|x|倍|万|亿)$",
    re.IGNORECASE,
)
_NUM_TOK_RE = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(%?)$")
_NEG_IDIOM_RE = re.compile(
    r"\b(?:not\s+only|not\s+just|whether\s+or\s+not)\b|不仅|不止|不只是|不光",
    re.IGNORECASE,
)
_NEG_POLARITY_RE = re.compile(
    r"\b(?:not|never|no\s+longer|cannot|can't|doesn't|don't|didn't|won't|isn't|aren't|wasn't|weren't|without|unsupported)\b|"
    r"不支持|无法|不能|并非|不是|禁止|未提供|不包含|不再支持",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+|\n+")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has",
    "had", "but", "into", "over", "under", "about", "after", "before", "between",
    "can", "may", "will", "would", "could", "should", "using", "used", "via", "per",
    "also", "such", "more", "most", "some", "any", "all", "its", "their", "our", "your",
    "http", "https", "www", "com", "org", "net", "html", "see", "ref", "link",
    "支持", "提供", "通过", "使用", "可以", "用于", "参考", "详见", "以及", "或者", "并且", "已经",
}


class ClaimInput(BaseModel):
    claim: str
    url: str


class ClaimVerificationItem(BaseModel):
    claim: str
    url: str
    verdict: str  # supported | partial | unsupported | unreachable | blocked
    score: float = 0.0
    anchor_coverage: float = 1.0
    term_coverage: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)
    page_title: str = ""
    status_code: int | None = None
    reason_code: str | None = None
    reason: str | None = None


class VerificationReport(BaseModel):
    version: str = __version__
    checked_at: str
    total_claims: int
    unique_urls: int
    reachable_urls: int
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    unreachable_count: int = 0
    blocked_count: int = 0
    citation_validity_rate: float = 0.0
    grounding_rate: float = 0.0
    elapsed_ms: int = 0
    untrusted_content: bool = True
    items: list[ClaimVerificationItem] = Field(default_factory=list)
    markdown: str | None = None


VerifierFetcherFn = Callable[..., Awaitable[FetchResult]]


def _strip_trailing_punct(url: str) -> str:
    return url.rstrip(".,;:!?。，；：！？）】』'\"")


def extract_claims_from_markdown(markdown: str, max_claims: int = 30) -> list[ClaimInput]:
    """从 Markdown 文本中提取包含公网 HTTP(S) 引用的主张句，自动跳过围栏代码块。"""
    claims: list[ClaimInput] = []
    seen_pairs: set[tuple[str, str]] = set()
    in_fence = False

    for raw_line in (markdown or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if stripped.startswith("#") and "http" not in stripped:
            continue

        md_matches = list(_MD_LINK_RE.finditer(stripped))
        found_urls: list[str] = []
        for m in md_matches:
            u = _strip_trailing_punct(m.group(2).strip())
            if u and u not in found_urls:
                found_urls.append(u)

        # 去掉已匹配的 Markdown 链接语法，保留可见锚文本，再扫描裸 URL
        plain = _MD_LINK_RE.sub(lambda m: m.group(1), stripped)
        for m in _BARE_URL_RE.finditer(plain):
            u = _strip_trailing_punct(m.group(1).strip())
            if u and u not in found_urls:
                found_urls.append(u)

        if not found_urls:
            continue

        claim_text = _BARE_URL_RE.sub("", plain)
        claim_text = re.sub(r"^[\-\*\+\d\.\)\s>]+", "", claim_text)
        claim_text = re.sub(r"[`*_]", "", claim_text)
        claim_text = re.sub(r"\s+", " ", claim_text).strip(" ()[]（）【】·-:：;；。")
        if len(claim_text) < 4:
            claim_text = stripped[:200]

        for u in found_urls:
            key = (claim_text.lower(), u)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            claims.append(ClaimInput(claim=claim_text[:400], url=u))
            if len(claims) >= max_claims:
                return claims
    return claims


def _extract_features(claim: str) -> tuple[list[str], list[str]]:
    """提取硬事实锚点（版本号/CVE/数字/大写缩写）与语义关键词。"""
    anchors: list[str] = []
    seen_a: set[str] = set()
    for m in _HARD_ANCHOR_RE.finditer(claim):
        tok = m.group(0).strip()
        low = tok.lower()
        if low not in seen_a and low not in _STOPWORDS:
            seen_a.add(low)
            anchors.append(tok)

    terms: list[str] = []
    seen_t: set[str] = set()
    for m in _EN_WORD_RE.finditer(claim):
        tok = m.group(0).strip("._-/")
        low = tok.lower()
        if len(low) >= 3 and low not in _STOPWORDS and low not in seen_t:
            seen_t.add(low)
            terms.append(tok)
    for m in _CJK_SPAN_RE.finditer(claim):
        span = m.group(0)
        if span in _STOPWORDS:
            continue
        if len(span) <= 4:
            if span not in seen_t:
                seen_t.add(span)
                terms.append(span)
        else:
            # 步长 1 滑动 2-gram，避免中文奇数偏移错位导致假阴性
            added = 0
            for i in range(len(span) - 1):
                bg = span[i : i + 2]
                if bg[0] in _CJK_EDGE_STOP or bg[-1] in _CJK_EDGE_STOP:
                    continue
                if len(bg) == 2 and bg not in _STOPWORDS and bg not in seen_t:
                    seen_t.add(bg)
                    terms.append(bg)
                    added += 1
                    if added >= 12:
                        break
    return anchors, terms


def _token_in_text(tok: str, text_low: str) -> bool:
    low = tok.strip().lower()
    if not low or not text_low:
        return False
    # 带单位数值（如 12ms / 64MB / 3x / 5万）：兼容可选空格并强制左侧非数字边界
    unit_m = _UNIT_NUM_RE.match(low)
    if unit_m:
        num, unit = unit_m.group(1), unit_m.group(2)
        if _CJK_CHAR_RE.search(unit):
            return re.search(rf"(?<![0-9.,]){re.escape(num)}\s*{re.escape(unit)}", text_low) is not None
        return re.search(rf"(?<![0-9.,]){re.escape(num)}\s*{re.escape(unit)}(?![a-z0-9_])", text_low) is not None

    # 中文无空格分词，保留子串匹配
    if _CJK_CHAR_RE.search(low):
        return low in text_low

    # 版本号（如 v2.0 / 2.0.5）：兼容可选 v 前缀，拒绝前缀/后缀数字或更长版本号子串假阳性
    ver_m = _VERSION_TOK_RE.match(low)
    if ver_m:
        ver = re.escape(ver_m.group(1))
        return re.search(rf"(?<![a-z0-9_.])v?{ver}(?![0-9]|\.[0-9])", text_low) is not None

    # 数字与百分比（兼容千分位 1000 <-> 1,000，拒绝 50% 命中 150% / 2025 命中 20250）
    num_m = _NUM_TOK_RE.match(low)
    if num_m:
        raw_num, pct = num_m.group(1), num_m.group(2)
        plain = raw_num.replace(",", "")
        variants = [raw_num]
        if plain not in variants:
            variants.append(plain)
        if plain.isdigit() and len(plain) >= 4:
            grouped = f"{int(plain):,}"
            if grouped not in variants:
                variants.append(grouped)
        alt = "|".join(re.escape(v) for v in variants)
        if pct:
            return re.search(rf"(?<![0-9.,])(?:{alt})%", text_low) is not None
        return re.search(rf"(?<![0-9.,])(?:{alt})(?![0-9]|(?:\.[0-9])|(?:,[0-9]))", text_low) is not None

    # CVE / 大写缩写 / 英文词：强制 ASCII 词边界，防止 rag 命中 fragmentation 等子串假阳性
    return re.search(rf"(?<![a-z0-9_]){re.escape(low)}(?![a-z0-9_])", text_low) is not None


def _has_negation(text: str) -> bool:
    cleaned = _NEG_IDIOM_RE.sub(" ", text or "")
    return _NEG_POLARITY_RE.search(cleaned) is not None


def _has_polarity_conflict(claim: str, content: str, matched_tokens: list[str]) -> bool:
    """在最佳支撑句窗内检测中英文否定极性反转（防止 'NOT support' 命中 'supports' 误判 supported）。"""
    if not content or not matched_tokens:
        return False
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(content) if len(s.strip()) >= 8]
    if not sents:
        sents = [content.strip()]
    min_hits = max(1, (len(matched_tokens) + 1) // 2)
    best_hits = 0
    scored: list[tuple[int, str]] = []
    for sent in sents[:200]:
        s_low = sent.lower()
        hits = sum(1 for t in matched_tokens if _token_in_text(t, s_low))
        if hits >= min_hits:
            scored.append((hits, sent))
            if hits > best_hits:
                best_hits = hits
    if not scored:
        return False
    top_sents = [s for h, s in scored if h == best_hits]
    claim_neg = _has_negation(claim)
    if claim_neg:
        return all(not _has_negation(s) for s in top_sents)
    return all(_has_negation(s) for s in top_sents)


def _pick_supporting_quotes(content: str, all_tokens: list[str], max_quotes: int = 2) -> list[str]:
    if not content or not all_tokens:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(content) if len(s.strip()) >= 15]
    if not sents:
        sents = [content[:260].strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(sents[:200]):
        s_low = sent.lower()
        hits = sum(1 for t in all_tokens if _token_in_text(t, s_low))
        if hits > 0:
            scored.append((hits, -idx, sent[:280]))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s for _, _, s in scored[:max_quotes]]


def _evaluate_claim_against_page(claim: str, url: str, page: FetchResult) -> ClaimVerificationItem:
    safe_u = redact_url_secrets(url)
    if not page.ok:
        verdict = "blocked" if page.reason_code == "ssrf-blocked" else "unreachable"
        return ClaimVerificationItem(
            claim=claim,
            url=safe_u,
            verdict=verdict,
            score=0.0,
            anchor_coverage=0.0,
            term_coverage=0.0,
            page_title=page.title,
            status_code=page.status_code,
            reason_code=page.reason_code,
            reason=page.reason,
        )

    haystack = f"{page.title}\n{page.content}".lower()
    anchors, terms = _extract_features(claim)
    matched_anchors = [a for a in anchors if _token_in_text(a, haystack)]
    missing_anchors = [a for a in anchors if not _token_in_text(a, haystack)]
    matched_terms = [t for t in terms if _token_in_text(t, haystack)]
    missing_terms = [t for t in terms if not _token_in_text(t, haystack)]

    anchor_cov = (len(matched_anchors) / len(anchors)) if anchors else 1.0
    term_cov = (len(matched_terms) / len(terms)) if terms else (anchor_cov if anchors else 0.0)

    # 综合评分：存在硬事实锚点时按 0.5*anchor + 0.5*term 加权
    score = round((0.5 * anchor_cov + 0.5 * term_cov) if anchors else term_cov, 4)

    # 合并去重后的已命中与缺失项（硬锚点排前便于一眼看出版本/数字幻觉）
    all_matched: list[str] = []
    for t in matched_anchors + matched_terms:
        if t.lower() not in {x.lower() for x in all_matched}:
            all_matched.append(t)
    all_missing: list[str] = []
    for t in missing_anchors + missing_terms:
        if t.lower() not in {x.lower() for x in all_missing}:
            all_missing.append(t)

    reason_code: str | None = None
    reason: str | None = None
    if score >= 0.55 and (not anchors or anchor_cov >= 0.5):
        verdict = "supported"
    elif score >= 0.25 or (term_cov >= 0.4 and anchors and anchor_cov < 0.5):
        verdict = "partial"
    else:
        verdict = "unsupported"

    if verdict in ("supported", "partial") and _has_polarity_conflict(claim, page.content, all_matched):
        verdict = "unsupported"
        score = min(score, 0.15)
        reason_code = "polarity-conflict"
        reason = "claim polarity conflicts with matched source statement"

    quotes = _pick_supporting_quotes(page.content, all_matched or terms or anchors, max_quotes=2)
    return ClaimVerificationItem(
        claim=claim,
        url=safe_u,
        verdict=verdict,
        score=score,
        anchor_coverage=round(anchor_cov, 4),
        term_coverage=round(term_cov, 4),
        matched_terms=all_matched[:12],
        missing_terms=all_missing[:12],
        supporting_quotes=quotes,
        page_title=page.title,
        status_code=page.status_code,
        reason_code=reason_code,
        reason=reason,
    )


def render_verification_markdown(report: VerificationReport) -> str:
    lines = [
        "# 引用核验审计报告",
        "",
        f"- **核验时间**：`{report.checked_at}` · **耗时**：`{report.elapsed_ms} ms`",
        f"- **总体统计**：主张 `{report.total_claims}` 条（独立 URL `{report.unique_urls}`，可达 `{report.reachable_urls}`）",
        f"- **链接存活率**：`{report.citation_validity_rate:.1%}` · **强支撑率**：`{report.grounding_rate:.1%}`",
        f"- **判定分布**：`supported={report.supported_count}` / `partial={report.partial_count}` / `unsupported={report.unsupported_count}` / `unreachable={report.unreachable_count}` / `blocked={report.blocked_count}`",
        "",
        "| # | 判定 | 对齐分 | 主张摘要 | 引用 URL | 缺失项 / 原因 |",
        "|---:|---|---:|---|---|---|",
    ]
    for idx, it in enumerate(report.items, start=1):
        claim_short = it.claim.replace("|", "\\|")[:80]
        detail = (
            ", ".join(it.missing_terms[:5])
            if it.verdict in ("partial", "unsupported") and it.missing_terms
            else (it.reason or it.reason_code or "-")
        ).replace("|", "\\|")
        lines.append(f"| {idx} | `{it.verdict}` | `{it.score:.2f}` | {claim_short} | {it.url} | {detail} |")
    lines.append("")
    return "\n".join(lines)


async def verify_claims(
    claims: list[ClaimInput] | None = None,
    markdown: str | None = None,
    concurrency: int = 5,
    timeout_sec: float = 15.0,
    max_chars: int = 12000,
    disk_cache: Any | None = None,
    use_cache: bool = True,
    render_markdown: bool = False,
    fetcher: VerifierFetcherFn | None = None,
) -> VerificationReport:
    """并发核验一组主张或 Markdown 报告中的全部外链引用。"""
    t0 = time.perf_counter()
    target_claims: list[ClaimInput] = list(claims or [])
    if markdown:
        target_claims.extend(extract_claims_from_markdown(markdown))

    # 去重独立 URL，并发抓取一次复用给所有引用该 URL 的主张
    unique_urls: list[str] = []
    seen_u: set[str] = set()
    for c in target_claims:
        u = c.url.strip()
        if u and u not in seen_u:
            seen_u.add(u)
            unique_urls.append(u)

    eff_conc = max(1, min(10, int(concurrency)))
    sem = asyncio.Semaphore(eff_conc)

    async def _fetch_one(u: str) -> tuple[str, FetchResult]:
        async with sem:
            if fetcher is not None:
                res = await fetcher(url=u, max_chars=max_chars, timeout_sec=timeout_sec)
            else:
                res = await safe_fetch_url(
                    url=u,
                    max_chars=max_chars,
                    timeout_sec=timeout_sec,
                    disk_cache=disk_cache,
                    use_cache=use_cache,
                )
            return u, res

    fetched_pairs = await asyncio.gather(*(_fetch_one(u) for u in unique_urls)) if unique_urls else []
    page_map = dict(fetched_pairs)

    items: list[ClaimVerificationItem] = []
    for c in target_claims:
        page = page_map.get(c.url.strip()) or FetchResult(
            ok=False,
            url=redact_url_secrets(c.url),
            reason_code="error",
            reason="missing fetch result",
        )
        items.append(_evaluate_claim_against_page(c.claim, c.url, page))

    counts = {"supported": 0, "partial": 0, "unsupported": 0, "unreachable": 0, "blocked": 0}
    for it in items:
        counts[it.verdict] = counts.get(it.verdict, 0) + 1

    reachable_urls = sum(1 for res in page_map.values() if res.ok)
    total = len(items)
    valid_rate = round(reachable_urls / len(unique_urls), 4) if unique_urls else 0.0
    ground_rate = round(counts["supported"] / total, 4) if total else 0.0
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    report = VerificationReport(
        version=__version__,
        checked_at=datetime.now(timezone.utc).isoformat(),
        total_claims=total,
        unique_urls=len(unique_urls),
        reachable_urls=reachable_urls,
        supported_count=counts["supported"],
        partial_count=counts["partial"],
        unsupported_count=counts["unsupported"],
        unreachable_count=counts["unreachable"],
        blocked_count=counts["blocked"],
        citation_validity_rate=valid_rate,
        grounding_rate=ground_rate,
        elapsed_ms=elapsed_ms,
        untrusted_content=True,
        items=items,
    )
    if render_markdown:
        report.markdown = render_verification_markdown(report)
    return report
