"""research_evidence: 跨 Lane 候选融合、主机多样性挑选与失败顺延原文核验模块。"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlsplit

from omni_search.contracts import SearchItem, SearchResponse
from omni_search.pool import normalize_url_key
from omni_search.research_audit import (
    classify_evidence_tier,
    compute_confidence,
    extract_pub_year,
    extract_relevant_quotes,
)
from omni_search.research_models import FetcherFn, VerifiedEvidence
from omni_search.research_planner import _compute_query_relevance


def _merge_lane_candidates(
    lane_pairs: list[tuple[str, SearchResponse]],
    max_candidates: int = 35,
    brief: str = "",
) -> list[SearchItem]:
    """跨 lane 按规范化 URL 合并候选，融合多 lane 印证、查询相关性重排与条件化权威度分级（防泛词 T1 霸榜）。"""
    bucket: dict[str, tuple[float, SearchItem, list[str], list[str], dict[str, int]]] = {}
    for lane_name, resp in lane_pairs:
        for rank, item in enumerate(resp.results, start=1):
            key = normalize_url_key(item.url) or f"{item.domain}:{item.title.lower()}"
            delta = (item.score if item.score > 0 else 1.0 / (60 + rank)) + 0.01 / (1 + rank)
            item_sources = list(item.extra.get("matched_sources") or [item.source])
            yr = extract_pub_year(item.published_at, item.url, item.snippet)

            if key in bucket:
                prev_score, prev_item, lanes, sources, src_years = bucket[key]
                if lane_name not in lanes:
                    lanes.append(lane_name)
                for s in item_sources:
                    if s not in sources:
                        sources.append(s)
                if yr is not None:
                    src_years[item.source] = yr
                if len(item.snippet) > len(prev_item.snippet):
                    prev_item.snippet = item.snippet
                if not prev_item.published_at and item.published_at:
                    prev_item.published_at = item.published_at
                bucket[key] = (prev_score + delta + 0.02, prev_item, lanes, sources, src_years)
            else:
                copied = item.model_copy(deep=True)
                init_years = {item.source: yr} if yr is not None else {}
                bucket[key] = (delta, copied, [lane_name], item_sources, init_years)

    rescored: list[tuple[float, float, SearchItem, list[str], list[str], dict[str, int]]] = []
    for base_score, item, lanes, sources, src_years in bucket.values():
        tier = classify_evidence_tier(item.url, item.source)
        tier_bonus = 0.006 if tier == "T1" else (0.003 if tier == "T2" else 0.0)
        rel, anchor_cnt = _compute_query_relevance(item, brief) if brief else (1.0, 0)
        if brief and anchor_cnt >= 3:
            if rel < 0.18:
                final_score = base_score * 0.40
            else:
                final_score = base_score + tier_bonus + 0.035 * rel
        else:
            final_score = base_score + tier_bonus
        rescored.append((final_score, rel, item, lanes, sources, src_years))

    merged: list[SearchItem] = []
    seen_repo_slugs: dict[str, SearchItem] = {}
    for score, rel, item, lanes, sources, src_years in sorted(rescored, key=lambda x: x[0], reverse=True):
        if item.domain == "code" and "/" in item.title:
            slug = item.title.rsplit("/", 1)[-1].strip().lower()
            if len(slug) >= 4:
                if slug in seen_repo_slugs:
                    primary = seen_repo_slugs[slug]
                    primary.extra["duplicate_variants"] = int(primary.extra.get("duplicate_variants", 0)) + 1
                    continue
                seen_repo_slugs[slug] = item

        tier = classify_evidence_tier(item.url, item.source)
        item.score = round(score, 6)
        item.extra["matched_lanes"] = lanes
        item.extra["matched_sources"] = sources
        item.extra["evidence_tier"] = tier
        if brief:
            item.extra["relevance"] = round(rel, 3)
        if src_years:
            item.extra["source_years"] = src_years
        item.extra["confidence"] = compute_confidence(tier, len(lanes), len(sources), verified=False)
        merged.append(item)
        if len(merged) >= max_candidates:
            break
    return merged


def _is_low_value_redirect_url(url: str) -> bool:
    """识别聚合跳转/加密中转链接（如 Google News RSS 加密跳转链），避免浪费原文核验配额。"""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host == "news.google.com" and parsed.path.startswith("/rss/articles/"):
        return True
    return False


def _select_diverse_verify_targets(candidates: list[SearchItem], top_k: int) -> list[SearchItem]:
    """按主机名多样性优先挑选 Top-K 核验目标（每站点最多先选 1 条，优先直连原文与高权威度）。"""
    if top_k <= 0:
        return []
    http_cands = [
        c
        for c in candidates
        if c.url.startswith(("http://", "https://")) and bool((urlsplit(c.url).hostname or "").strip())
    ]
    picked: list[SearchItem] = []
    seen_hosts: set[str] = set()
    overflow: list[SearchItem] = []

    prioritized = sorted(
        http_cands,
        key=lambda c: (
            1 if _is_low_value_redirect_url(c.url) else 0,
            1 if float(c.extra.get("relevance", 1.0)) < 0.20 else 0,
            0 if c.extra.get("evidence_tier") == "T1" else (1 if c.extra.get("evidence_tier") == "T2" else 2),
            -c.score,
        ),
    )
    for c in prioritized:
        host = (urlsplit(c.url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in seen_hosts:
            seen_hosts.add(host)
            picked.append(c)
            if len(picked) >= top_k:
                return picked
        else:
            overflow.append(c)

    for c in overflow:
        if len(picked) >= top_k:
            break
        picked.append(c)
    return picked


async def _verify_candidates_with_fallback(
    candidates: list[SearchItem],
    brief: str,
    verify_top_k: int,
    deadline: float,
    fetch_fn: FetcherFn,
) -> tuple[list[VerifiedEvidence], set[str], list[str], bool]:
    """跨主机多样性原文核验 + 失败自动顺延补位 + Quote Grounding 证据句抽取。"""
    if verify_top_k <= 0:
        return [], set(), [], False
    queue = _select_diverse_verify_targets(candidates, verify_top_k + 3)
    if not queue:
        return [], set(), [], False

    verified_evidence: list[VerifiedEvidence] = []
    verified_urls: set[str] = set()
    gaps: list[str] = []
    budget_truncated = False

    first_batch = queue[:verify_top_k]
    fallback_batch = queue[verify_top_k:]

    async def _fetch_one(cand: SearchItem, timeout_s: float) -> VerifiedEvidence:
        tier = str(cand.extra.get("evidence_tier") or "T3")
        try:
            fres = await asyncio.wait_for(
                fetch_fn(cand.url, max_chars=3000, timeout_sec=timeout_s),
                timeout=timeout_s + 0.5,
            )
        except asyncio.TimeoutError:
            gaps.append(f"verify timeout url={cand.url} > {timeout_s:.1f}s")
            return VerifiedEvidence(
                url=cand.url,
                title=cand.title,
                evidence_tier=tier,
                verified=False,
                reason_code="deadline",
                reason=f"verify timeout > {timeout_s:.1f}s",
            )
        except Exception as exc:
            gaps.append(f"verify failed url={cand.url} err={exc}")
            return VerifiedEvidence(
                url=cand.url,
                title=cand.title,
                evidence_tier=tier,
                verified=False,
                reason_code="error",
                reason=str(exc),
            )
        if fres.ok:
            verified_urls.add(normalize_url_key(cand.url))
            sigs = getattr(fres, "content_signals", None) or {}
            restricted = [f"{k}={sigs[k]}" for k in ("ai-input", "search") if sigs.get(k) == "no"]
            if restricted:
                gaps.append(f"content-signal-restricted: url={cand.url} ({', '.join(restricted)})")
            quotes = extract_relevant_quotes(fres.content, brief, max_quotes=2)
            return VerifiedEvidence(
                url=cand.url,
                title=fres.title or cand.title,
                evidence_tier=tier,
                verified=True,
                status_code=fres.status_code,
                excerpt=fres.content[:600],
                relevant_quotes=quotes,
                untrusted_content=fres.untrusted_content,
            )
        gaps.append(f"verify failed url={cand.url} code={fres.reason_code} ({fres.reason})")
        return VerifiedEvidence(
            url=cand.url,
            title=cand.title,
            evidence_tier=tier,
            verified=False,
            status_code=fres.status_code,
            reason_code=fres.reason_code,
            reason=fres.reason,
        )

    rem_sec = deadline - time.perf_counter()
    if rem_sec <= 1.0:
        return [], set(), ["verify stage skipped due to budget"], True

    per_timeout = min(10.0, max(2.0, rem_sec - 0.5))
    first_res = list(await asyncio.gather(*(_fetch_one(c, per_timeout) for c in first_batch)))
    verified_evidence.extend(first_res)

    ok_count = sum(1 for v in verified_evidence if v.verified)
    for fb_cand in fallback_batch:
        if ok_count >= verify_top_k:
            break
        rem_sec = deadline - time.perf_counter()
        if rem_sec <= 1.5:
            budget_truncated = True
            break
        ev = await _fetch_one(fb_cand, min(8.0, rem_sec - 0.5))
        verified_evidence.append(ev)
        if ev.verified:
            ok_count += 1

    return verified_evidence, verified_urls, gaps, budget_truncated
