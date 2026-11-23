"""asyncio 联邦并发执行池：预算、单源截止、熔断探测与 RRF 跨源去重融合。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from omni_search import __version__
from omni_search.cache import DiskCacheStore
from omni_search.contracts import (
    CompletenessReport,
    EngineEnvelope,
    EngineFailure,
    LedgerSummary,
    ReasonCode,
    RequestStats,
    SearchItem,
    SearchResponse,
    build_coverage,
)
from omni_search.health import HealthStore
from omni_search.router import route_domains

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
}


class SearchEngine(Protocol):
    id: str
    domain: str
    tier: str
    deadline_sec: float

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope: ...


_ARXIV_CANON_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|doi\.org/10\.48550/arxiv\.)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


def normalize_url_key(url: str) -> str:
    """URL 规范化用于跨源去重（统一 arXiv/DOI 标识、剥除追踪参数、尾部斜杠与 fragment）。"""
    if not url:
        return ""
    raw = url.strip()
    am = _ARXIV_CANON_RE.search(raw)
    if am:
        return f"arxiv:{am.group(1)}"
    try:
        p = urlsplit(raw)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path.rstrip("/") or "/"
        qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() not in _TRACKING_PARAMS]
        qs.sort()
        return urlunsplit((p.scheme.lower() or "https", host, path, urlencode(qs), ""))
    except Exception:
        return raw.lower()


_POOL_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "how",
    "what", "why", "when", "where", "which", "about", "your", "our", "are", "was",
    "were", "not", "can", "will", "use", "using", "based", "build", "make", "best",
    "top", "new", "guide", "tutorial", "list", "vs", "versus",
    "to", "of", "in", "on", "at", "by", "an", "is", "be", "do", "as", "or", "if",
}

_POOL_GENERIC_TERMS = {
    "ai", "llm", "web", "app", "api", "tool", "tools", "system", "platform", "service",
    "server", "client", "code", "data", "open", "source", "project", "framework",
}


def _query_terms(query: str) -> list[str]:
    if not query:
        return []
    toks = [
        t
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", query.lower())
        if len(t) >= 2 and t not in _POOL_STOPWORDS
    ]
    return list(dict.fromkeys(toks))[:10]


def relax_long_query(query: str, max_terms: int = 4) -> str:
    """将长自然语言查询提炼为高信息量核心宽词（优先保留专有名词/缩写/带符号词，跳过纯年份与停用词）。"""
    raw_toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", query or "")
    seen: set[str] = set()
    scored: list[tuple[float, int, str]] = []
    for idx, tok in enumerate(raw_toks):
        low = tok.lower()
        if len(low) < 2 or low in _POOL_STOPWORDS or low in seen:
            continue
        if re.fullmatch(r"20\d{2}|19\d{2}", low):
            continue
        seen.add(low)
        score = 1.0
        if any(ch in tok for ch in ("-", "_", ".", "+")):
            score += 2.2
        if tok.isupper() and len(tok) >= 2:
            score += 1.8
        elif any(c.isupper() for c in tok[1:]):
            score += 1.4
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            score += 1.6
        if len(low) >= 6:
            score += 0.6
        if low in _POOL_GENERIC_TERMS:
            score -= 0.9
        scored.append((score, idx, tok))
    if not scored:
        return " ".join(raw_toks[:max_terms])
    picked = sorted(scored, key=lambda x: (-x[0], x[1]))[: max(1, max_terms)]
    picked.sort(key=lambda x: x[1])
    return " ".join(t for _, _, t in picked)


_RECENCY_DOMAINS = {"news", "security"}
_YEAR_SIMPLE_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def _recency_bonus(item: SearchItem, current_year: int | None = None) -> float:
    """对时效敏感域（news / security）施加温和时间新鲜度加权，不惩罚无日期条目及经典知识域。"""
    if item.domain not in _RECENCY_DOMAINS or not item.published_at:
        return 0.0
    m = _YEAR_SIMPLE_RE.search(str(item.published_at))
    if not m:
        return 0.0
    now_yr = current_year or datetime.now(timezone.utc).year
    age = now_yr - int(m.group(1))
    if age <= 1:
        return 0.0045
    if age <= 3:
        return 0.0015
    if age >= 6:
        return -0.0035
    return 0.0


_SOURCE_QUALITY_WEIGHT: dict[str, float] = {
    "github_direct": 1.05,
    "arxiv": 1.05,
    "nvd_cve": 1.08,
    "osv_vulns": 1.08,
    "github_issues": 0.78,
    "google_news_rss": 0.85,
}


def fuse_results_rrf(envelopes: list[EngineEnvelope], k: int = 60, query: str = "") -> list[SearchItem]:
    """跨源 Reciprocal Rank Fusion 去重融合 + 信噪比先验 + 标题/摘要命中加成 + 零重合噪声降权 + 时效鲜度加权（不删结果）。"""
    bucket: dict[str, tuple[float, SearchItem, list[str]]] = {}
    qterms = _query_terms(query)
    for env in envelopes:
        src_w = _SOURCE_QUALITY_WEIGHT.get(env.engine, 1.0)
        for rank, item in enumerate(env.results, start=1):
            if item.url and item.url.lower().startswith(("http://", "https://")) and not urlsplit(item.url).hostname:
                continue
            key = normalize_url_key(item.url) or f"{item.domain}:{item.title.lower()}"
            rrf_delta = src_w / (k + rank)
            if key in bucket:
                prev_score, prev_item, sources = bucket[key]
                if env.engine not in sources:
                    sources.append(env.engine)
                if len(item.snippet) > len(prev_item.snippet):
                    prev_item.snippet = item.snippet
                if not prev_item.published_at and item.published_at:
                    prev_item.published_at = item.published_at
                bucket[key] = (prev_score + rrf_delta, prev_item, sources)
            else:
                copied = item.model_copy(deep=True)
                bucket[key] = (rrf_delta, copied, [env.engine])

    now_yr = datetime.now(timezone.utc).year
    scored: list[tuple[float, SearchItem, list[str]]] = []
    for base_score, item, sources in bucket.values():
        boost = 0.0
        zero_overlap = False
        if qterms:
            title_low = item.title.lower()
            snip_low = item.snippet.lower()
            url_low = item.url.lower()
            title_hits = sum(1 for t in qterms if t in title_low)
            snip_hits = sum(1 for t in qterms if t in snip_low)
            url_hits = sum(1 for t in qterms if t in url_low)
            boost = (0.008 * title_hits + 0.002 * snip_hits) / len(qterms)
            if (
                len(qterms) >= 2
                and item.domain not in ("local", "local_code")
                and (title_hits + snip_hits + url_hits) == 0
            ):
                zero_overlap = True
        rec = _recency_bonus(item, current_year=now_yr)
        total_score = base_score + boost + rec
        if zero_overlap:
            total_score *= 0.35
            item.extra["low_lexical_overlap"] = True
        scored.append((total_score, item, sources))

    ranked: list[SearchItem] = []
    for score, item, sources in sorted(scored, key=lambda x: x[0], reverse=True):
        item.score = round(score, 6)
        item.extra["matched_sources"] = sources
        ranked.append(item)
    return ranked


class FederatedPool:
    """多源并发调度池：强制单源超时、总预算、健康熔断、TTL 缓存与诚实覆盖率。"""

    def __init__(
        self,
        engines: list[SearchEngine],
        health_store: HealthStore | None = None,
        disabled_engines: set[str] | None = None,
        cache_ttl_sec: float = 120.0,
        reranker: Any | None = None,
        disk_cache: DiskCacheStore | None = None,
        telemetry: Any | None = None,
    ):
        self.engines = engines
        self.health_store = health_store or HealthStore()
        self.reranker = reranker
        self.telemetry = telemetry
        cache_off = os.environ.get("OMNI_CACHE", "1").strip().lower() in ("0", "false", "off")
        self.cache_ttl_sec = 0.0 if cache_off else max(0.0, float(cache_ttl_sec))
        self.disk_cache = None if cache_off else disk_cache
        self._cache: dict[tuple[Any, ...], tuple[float, SearchResponse]] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None
        if disabled_engines is None:
            raw = os.environ.get("OMNI_DISABLE", "")
            self.disabled_engines = {x.strip().lower() for x in raw.replace(";", ",").split(",") if x.strip()}
        else:
            self.disabled_engines = {x.lower() for x in disabled_engines}

    def _get_http_client(self) -> httpx.AsyncClient:
        """懒加载复用持久 httpx.AsyncClient（跨多轮复用 TCP/TLS 连接池，事件循环切换时自动重建）。"""
        loop = asyncio.get_running_loop()
        if self._http_client is None or self._http_client.is_closed or self._http_loop is not loop:
            self._http_client = httpx.AsyncClient(
                follow_redirects=True,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=20, keepalive_expiry=30.0),
            )
            self._http_loop = loop
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
        self._http_client = None
        self._http_loop = None

    async def _run_one(
        self,
        eng: SearchEngine,
        query: str,
        limit: int,
        client: httpx.AsyncClient,
        budget_deadline: float,
    ) -> EngineEnvelope:
        t0 = time.perf_counter()
        cred_fp = eng.credential_fingerprint() if hasattr(eng, "credential_fingerprint") else ""
        health = self.health_store.check(eng.id, cred_fingerprint=cred_fp)
        eff_deadline = eng.deadline_sec
        warnings: list[str] = []
        if health.auto_healed:
            warnings.append(f"[{eng.id}] 检测到凭据变更，已自动解除历史熔断冷却")
        if health.skip_fast:
            try:
                rc = ReasonCode(health.last_reason) if health.last_reason else ReasonCode.DEADLINE
            except ValueError:
                rc = ReasonCode.DEADLINE
            rem_cd = max(1, int(health.cooldown_until - time.time()))
            msg = f"circuit-open fast-fail (last={rc.value}, fails={health.consecutive_failures}, cooldown={rem_cd}s)"
            warnings.append(f"[{eng.id}] 熔断快退（连续失败 {health.consecutive_failures} 次，上次原因={rc.value}，剩余冷却 {rem_cd}s），已跳过网络请求")
            return EngineEnvelope(
                engine=eng.id,
                domain=eng.domain,
                elapsed_ms=0,
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(reason_code=rc, reason=msg),
                warnings=warnings,
            )
        if (health.in_cooldown or health.half_open) and health.probe_deadline_sec > 0:
            eff_deadline = min(eff_deadline, health.probe_deadline_sec)
            phase_lbl = "冷却期内半开探测" if health.in_cooldown else "冷却到期半开探测"
            warnings.append(
                f"[{eng.id}] {phase_lbl}（连续失败 {health.consecutive_failures} 次，上次原因={health.last_reason}），截止收紧至 {eff_deadline:.1f}s"
            )

        remaining = budget_deadline - time.perf_counter()
        if remaining <= 0.05:
            return EngineEnvelope(
                engine=eng.id,
                domain=eng.domain,
                elapsed_ms=0,
                failure=EngineFailure(reason_code=ReasonCode.BUDGET_TRUNCATED, reason="pool budget exhausted before launch"),
                warnings=warnings,
            )
        timeout = min(eff_deadline, remaining)

        try:
            env = await asyncio.wait_for(eng.search(query, limit, client), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - t0) * 1000)
            rc = ReasonCode.BUDGET_TRUNCATED if remaining < eff_deadline else ReasonCode.DEADLINE
            msg = f"engine timeout after {timeout:.2f}s"
            self.health_store.record_failure(eng.id, rc, cred_fingerprint=cred_fp)
            warnings.append(f"[{eng.id}] {msg}")
            return EngineEnvelope(
                engine=eng.id,
                domain=eng.domain,
                elapsed_ms=elapsed,
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=rc, reason=msg),
                warnings=warnings,
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            self.health_store.record_failure(eng.id, ReasonCode.ERROR, cred_fingerprint=cred_fp)
            warnings.append(f"[{eng.id}] unexpected error: {exc}")
            return EngineEnvelope(
                engine=eng.id,
                domain=eng.domain,
                elapsed_ms=elapsed,
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=str(exc)),
                warnings=warnings,
            )

        env.warnings = warnings + list(env.warnings)
        if env.failure is None:
            self.health_store.record_success(eng.id)
        else:
            self.health_store.record_failure(eng.id, env.failure.reason_code, cred_fingerprint=cred_fp)
            env.warnings.append(f"[{eng.id}] {env.failure.reason_code.value}: {env.failure.reason}")
        return env

    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        limit: int = 6,
        max_results: int = 30,
        budget_ms: int = 25000,
        use_cache: bool = True,
        auto_relax: bool = True,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        run_id = uuid.uuid4().hex[:12]
        target_domains = route_domains(query, domains)
        has_local_dom = any(d in ("local", "local_code") for d in target_domains)
        rr_id = getattr(self.reranker, "id", "") if self.reranker is not None else ""
        cache_key = (query.strip().lower(), tuple(target_domains), int(limit), int(max_results), rr_id)
        disk_key = hashlib.sha256(
            json.dumps(
                [cache_key[0], list(cache_key[1]), cache_key[2], cache_key[3], sorted(self.disabled_engines), rr_id],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        if use_cache and self.cache_ttl_sec > 0 and cache_key in self._cache:
            exp_ts, cached_resp = self._cache[cache_key]
            if time.monotonic() < exp_ts:
                hit = cached_resp.model_copy(deep=True)
                hit.cached = True
                hit.ledger.elapsed_ms = 0
                return hit
            del self._cache[cache_key]

        if use_cache and not has_local_dom and self.disk_cache is not None:
            raw_disk = self.disk_cache.get(disk_key)
            if raw_disk:
                try:
                    hit = SearchResponse.model_validate(json.loads(raw_disk))
                    hit.cached = True
                    hit.ledger.elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    return hit
                except Exception:
                    pass

        selected: list[SearchEngine] = []
        disabled_matched: list[str] = []
        for eng in self.engines:
            if eng.domain not in target_domains:
                continue
            if eng.id.lower() in self.disabled_engines:
                disabled_matched.append(eng.id)
                continue
            selected.append(eng)

        budget_deadline = t0 + max(0.2, budget_ms / 1000.0)
        envelopes: list[EngineEnvelope] = []
        http_client = self._get_http_client()
        if selected:
            tasks = [
                self._run_one(eng, query, limit, http_client, budget_deadline)
                for eng in selected
            ]
            envelopes = list(await asyncio.gather(*tasks))

        relax_warnings: list[str] = []
        raw_word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", query or ""))
        has_local_filter = bool(re.search(r"\b(?:type|ext|collection|tag):", query or ""))
        if (
            auto_relax
            and not has_local_filter
            and raw_word_count >= 5
            and (budget_deadline - time.perf_counter()) > 2.0
        ):
            eng_by_id = {e.id: e for e in selected}
            zero_hit_idxs = [
                idx
                for idx, env in enumerate(envelopes)
                if env.failure is None
                and not env.results
                and env.domain not in ("local", "local_code")
                and env.engine in eng_by_id
            ]
            initial_hits = sum(len(env.results) for env in envelopes)
            executed_remote = sum(1 for env in envelopes if env.failure is None and env.domain not in ("local", "local_code"))
            if zero_hit_idxs and (initial_hits < max(3, limit) or (executed_remote >= 2 and len(zero_hit_idxs) * 2 >= executed_remote)):
                relaxed_q = relax_long_query(query, max_terms=4)
                if relaxed_q and relaxed_q.strip().lower() != query.strip().lower():
                    retry_tasks = [
                        self._run_one(eng_by_id[envelopes[i].engine], relaxed_q, limit, http_client, budget_deadline)
                        for i in zero_hit_idxs
                    ]
                    retry_envs = list(await asyncio.gather(*retry_tasks))
                    added_hits = 0
                    for idx, renv in zip(zero_hit_idxs, retry_envs):
                        orig = envelopes[idx]
                        merged_req = RequestStats(
                            attempted=orig.requests.attempted + renv.requests.attempted,
                            succeeded=orig.requests.succeeded + renv.requests.succeeded,
                            failed=orig.requests.failed + renv.requests.failed,
                        )
                        if renv.failure is None and renv.results:
                            added_hits += len(renv.results)
                            renv.requests = merged_req
                            renv.elapsed_ms = orig.elapsed_ms + renv.elapsed_ms
                            renv.warnings = list(orig.warnings) + list(renv.warnings)
                            envelopes[idx] = renv
                        else:
                            orig.requests = merged_req
                            orig.elapsed_ms += renv.elapsed_ms
                    relax_warnings.append(
                        f"[pool] 长查询宽词补查: '{relaxed_q}'（补查 {len(zero_hit_idxs)} 个 0 命中源，新增 {added_hits} 条候选）"
                    )

        coverage = build_coverage(envelopes, disabled_engines=disabled_matched)
        fused = fuse_results_rrf(envelopes, query=query)
        rerank_warnings: list[str] = []
        if self.reranker is not None and len(fused) > 1:
            rem_sec = budget_deadline - time.perf_counter()
            if rem_sec > 0.5:
                try:
                    fused = await asyncio.wait_for(
                        self.reranker.rerank(
                            query=query,
                            items=fused,
                            client=http_client,
                            top_n=max(max_results, 20) if max_results > 0 else 25,
                        ),
                        timeout=min(rem_sec, 10.0),
                    )
                except Exception as exc:
                    rr_name = getattr(self.reranker, "id", "reranker")
                    rerank_warnings.append(f"[{rr_name}] rerank failed: {exc}; kept RRF order")
            else:
                rr_name = getattr(self.reranker, "id", "reranker")
                rerank_warnings.append(f"[{rr_name}] rerank skipped due to budget; kept RRF order")

        total_candidates = len(fused)
        truncated = max_results > 0 and total_candidates > max_results
        final_results = fused[:max_results] if max_results > 0 else fused

        if self.telemetry is not None and envelopes:
            top5_counts: dict[str, int] = {}
            for item in final_results[:5]:
                for src in item.extra.get("matched_sources") or [item.source]:
                    top5_counts[str(src)] = top5_counts.get(str(src), 0) + 1
            tel_events = []
            for env in envelopes:
                rc = env.failure.reason_code.value if env.failure else ("zero-hit" if not env.results else "ok")
                is_fast_skip = bool(env.failure and env.failure.reason.startswith("circuit-open fast-fail"))
                tel_events.append(
                    {
                        "engine": env.engine,
                        "domain": env.domain,
                        "ok": env.failure is None,
                        "reason_code": rc,
                        "elapsed_ms": env.elapsed_ms,
                        "result_count": len(env.results),
                        "top5_hits": top5_counts.get(env.engine, 0),
                        "skipped_fast": is_fast_skip,
                    }
                )
            try:
                self.telemetry.record_batch(tel_events)
            except Exception:
                pass

        warnings: list[str] = list(relax_warnings) + list(rerank_warnings)
        req_att = req_ok = req_fail = 0
        budget_hit = False
        for env in envelopes:
            warnings.extend(env.warnings)
            req_att += env.requests.attempted
            req_ok += env.requests.succeeded
            req_fail += env.requests.failed
            if env.failure and env.failure.reason_code == ReasonCode.BUDGET_TRUNCATED:
                budget_hit = True

        # 0 命中且存在 blocked/degraded 时标记低置信否定（0 命中 != 不存在）
        low_conf_neg = (len(final_results) == 0) and (bool(coverage.blocked) or bool(coverage.degraded) or coverage.verdict in ("none", "no-engines"))

        resp = SearchResponse(
            version=__version__,
            query=query,
            domains=target_domains,
            results=final_results,
            coverage=coverage,
            completeness=CompletenessReport(
                returned_count=len(final_results),
                total_candidates=total_candidates,
                truncated=truncated,
                per_source_limit=limit,
            ),
            warnings=warnings,
            ledger=LedgerSummary(
                run_id=run_id,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                budget_ms=budget_ms,
                budget_truncated=budget_hit,
                requests_attempted=req_att,
                requests_succeeded=req_ok,
                requests_failed=req_fail,
            ),
            cached=False,
            low_confidence_negative=low_conf_neg,
        )
        # 仅缓存非空结果，避免瞬时故障/空命中污染后续调用
        if use_cache and self.cache_ttl_sec > 0 and final_results:
            if len(self._cache) >= 256:
                self._cache.clear()
            self._cache[cache_key] = (time.monotonic() + self.cache_ttl_sec, resp.model_copy(deep=True))
        if (
            use_cache
            and not has_local_dom
            and self.disk_cache is not None
            and final_results
            and coverage.verdict in ("full", "partial")
        ):
            self.disk_cache.set(disk_key, resp.model_dump_json())
        return resp
