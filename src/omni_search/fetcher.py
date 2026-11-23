"""omni_fetch: 带 SSRF 防护、流式体积上限与防注入边界的安全正文抽取器。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from omni_search import __version__
from omni_search.contracts import redact_url_secrets, sanitize_text
from omni_search.fetch_apis import (  # noqa: F401 - re-exported for backward compatibility
    _GITHUB_RESERVED_OWNERS,
    _HF_RESERVED_PREFIXES,
    _WIKI_HOST_RE,
    _try_fetch_official_api,
)
from omni_search.fetch_extract import (  # noqa: F401 - re-exported for backward compatibility
    _MD_TOP_NAV_RE,
    _SLUG_STOPWORDS,
    _count_slug_hits,
    _detect_text_encoding,
    _extract_html_title_and_body,
    _extract_markdown_title,
    _extract_path_slug_tokens,
    _extract_pdf_bytes,
    _looks_like_site_root_llms_fallback,
    _parse_agentic_headers,
)
from omni_search.fetch_security import (  # noqa: F401 - re-exported for backward compatibility
    _MAX_BYTES,
    _MAX_PDF_BYTES,
    _MAX_PDF_PAGES,
    _MAX_REDIRECTS,
    _fetch_stream_with_redirects,
    _is_forbidden_ip,
    _validate_target_url,
)
from omni_search.html_markdown import (  # noqa: F401 - re-exported for backward compatibility
    _BLOCK_TAGS,
    _CleanTextExtractor,
    _HEADING_PREFIX,
    _MAIN_TAGS,
    _MIXED_ENTITY_RE,
    _PRUNE_STOPWORDS,
    _SKIP_TAGS,
    _normalize_extracted_text,
    _unescape_html_entities,
    prune_content_for_query,
)


class FetchResult(BaseModel):
    ok: bool
    url: str
    final_url: str = ""
    status_code: int | None = None
    content_type: str = ""
    source_via: str = "direct"
    title: str = ""
    content: str = ""
    truncated: bool = False
    cached: bool = False
    pruned_by_query: bool = False
    fetched_at: str = ""
    untrusted_content: bool = True
    content_signals: dict[str, str] = Field(default_factory=dict)
    markdown_tokens: int | None = None
    reason_code: str | None = None
    reason: str | None = None


async def _try_fetch_llms_txt(
    url: str,
    safe_url: str,
    max_chars: int,
    timeout_sec: float,
    cli: httpx.AsyncClient,
    resolve_host: bool,
    now_iso: str,
) -> FetchResult | None:
    """对站点根路径优先探测标准 /llms.txt（失败自动回退 /llms-full.txt），均失败时平滑回退常规拉取。"""
    parts = urlsplit(url.strip())
    if parts.path not in ("", "/") or parts.query:
        return None
    for suffix, via in (("/llms.txt", "llms-txt"), ("/llms-full.txt", "llms-full-txt")):
        llms_url = f"{parts.scheme}://{parts.netloc}{suffix}"
        ok_u, _ = _validate_target_url(llms_url, resolve_host=resolve_host)
        if not ok_u:
            continue
        try:
            r = await _fetch_stream_with_redirects(
                cli,
                llms_url,
                headers={"User-Agent": f"search-all-in-one/{__version__}", "Accept": "text/plain, text/markdown"},
                timeout_sec=min(timeout_sec, 6.0),
                max_bytes=_MAX_BYTES,
                resolve_host=resolve_host,
                max_redirects=0,
            )
            if r.status_code != 200 or not r.content:
                continue
            ctype = r.headers.get("content-type", "").lower()
            if "html" in ctype:
                continue
            enc = _detect_text_encoding(r.content, ctype, r.encoding)
            raw_txt = r.content.decode(enc, errors="replace")
            if "<html" in raw_txt[:300].lower():
                continue
            body = sanitize_text(_normalize_extracted_text(raw_txt))
            if len(body) < 20:
                continue
            sig_map, md_toks = _parse_agentic_headers(r.headers)
            return FetchResult(
                ok=True,
                url=safe_url,
                final_url=llms_url,
                status_code=200,
                content_type=r.headers.get("content-type", "text/plain"),
                source_via=via,
                title=f"{parts.hostname or ''} {suffix}".strip(),
                content=body[:max_chars],
                truncated=len(body) > max_chars or r.size_capped,
                fetched_at=now_iso,
                untrusted_content=True,
                content_signals=sig_map,
                markdown_tokens=md_toks,
            )
        except Exception:
            continue
    return None


async def safe_fetch_url(
    url: str,
    max_chars: int = 8000,
    timeout_sec: float = 15.0,
    client: httpx.AsyncClient | None = None,
    resolve_host: bool = True,
    disk_cache: Any | None = None,
    use_cache: bool = True,
    prefer_llms_txt: bool = False,
    query: str = "",
) -> FetchResult:
    """安全拉取公网 URL（逐跳前置 SSRF 校验 + 流式限额 + 磁盘缓存 + Markdown 协商 + 官方 API + /llms.txt + 查询裁剪 + PDF 抽取），恒标 untrusted_content=True。"""
    now_iso = datetime.now(timezone.utc).isoformat()
    safe_url = redact_url_secrets(url)
    allowed, block_reason = _validate_target_url(url, resolve_host=resolve_host)
    if not allowed:
        return FetchResult(
            ok=False,
            url=safe_url,
            fetched_at=now_iso,
            reason_code="ssrf-blocked",
            reason=block_reason,
        )

    eff_query = (query or "").strip()
    cache_key = hashlib.sha256(
        f"fetch|{url.strip()}|{int(max_chars)}|{int(prefer_llms_txt)}|{eff_query.lower()}".encode("utf-8")
    ).hexdigest()
    if use_cache and disk_cache is not None:
        raw_cached = disk_cache.get(cache_key)
        if raw_cached:
            try:
                hit = FetchResult.model_validate(json.loads(raw_cached))
                hit.cached = True
                return hit
            except Exception:
                pass

    def _finalize_and_store(res: FetchResult, full_body: str | None = None) -> FetchResult:
        if res.ok:
            src_body = full_body if full_body is not None else res.content
            if eff_query:
                pruned_txt, did_prune = prune_content_for_query(src_body, eff_query, max_chars)
                if did_prune:
                    res.content = pruned_txt
                    res.pruned_by_query = True
                    res.truncated = True
                elif len(res.content) > max_chars:
                    res.content = res.content[:max_chars]
                    res.truncated = True
            elif len(res.content) > max_chars:
                res.content = res.content[:max_chars]
                res.truncated = True
        if res.ok and use_cache and disk_cache is not None:
            try:
                disk_cache.set(cache_key, res.model_dump_json())
            except Exception:
                pass
        return res

    ua = f"search-all-in-one/{__version__} (+https://github.com/omni-search)"
    headers = {
        "User-Agent": ua,
        "Accept": "text/markdown, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.8, application/pdf;q=0.7",
    }
    own_client = client is None
    cli = client or httpx.AsyncClient(follow_redirects=False)
    fetch_cap = min(60000, max(max_chars, max_chars * 4)) if eff_query else max_chars
    try:
        if prefer_llms_txt:
            llms_res = await _try_fetch_llms_txt(url, safe_url, fetch_cap, timeout_sec, cli, resolve_host, now_iso)
            if llms_res is not None:
                return _finalize_and_store(llms_res, full_body=llms_res.content)

        api_res = await _try_fetch_official_api(
            url,
            safe_url,
            fetch_cap,
            timeout_sec,
            cli,
            resolve_host,
            now_iso,
            validate_url=_validate_target_url,
            result_cls=FetchResult,
        )
        if api_res is not None:
            return _finalize_and_store(api_res, full_body=api_res.content)

        sresp = await _fetch_stream_with_redirects(
            cli,
            url,
            headers=headers,
            timeout_sec=timeout_sec,
            max_bytes=_MAX_BYTES,
            resolve_host=resolve_host,
            max_redirects=_MAX_REDIRECTS,
            allow_pdf_cap=True,
        )
        if sresp.redirect_blocked_url is not None:
            return FetchResult(
                ok=False,
                url=safe_url,
                final_url=redact_url_secrets(sresp.redirect_blocked_url),
                status_code=sresp.status_code,
                fetched_at=now_iso,
                reason_code="ssrf-blocked",
                reason=f"redirect blocked: {sresp.redirect_blocked_reason}",
            )
        if sresp.too_many_redirects:
            return FetchResult(
                ok=False,
                url=safe_url,
                final_url=redact_url_secrets(sresp.final_url),
                fetched_at=now_iso,
                reason_code="too-many-redirects",
                reason=f"exceeded {_MAX_REDIRECTS} redirects",
            )

        final_url_str = sresp.final_url
        if sresp.status_code >= 400:
            return FetchResult(
                ok=False,
                url=safe_url,
                final_url=redact_url_secrets(final_url_str),
                status_code=sresp.status_code,
                fetched_at=now_iso,
                reason_code="rate-limited" if sresp.status_code == 429 else "http-error",
                reason=f"HTTP {sresp.status_code}",
            )

        ctype = sresp.headers.get("content-type", "")
        ctype_low = ctype.lower()
        is_pdf = "application/pdf" in ctype_low or sresp.content[:5] == b"%PDF-"
        title = ""
        source_via = "direct"
        resp_headers = sresp.headers
        resp_status = sresp.status_code
        size_capped = sresp.size_capped

        if is_pdf:
            source_via = "pdf"
            try:
                title, body = await asyncio.to_thread(_extract_pdf_bytes, sresp.content[:_MAX_PDF_BYTES])
            except Exception as exc:
                return FetchResult(
                    ok=False,
                    url=safe_url,
                    final_url=redact_url_secrets(final_url_str),
                    status_code=resp_status,
                    content_type=ctype,
                    source_via="pdf",
                    fetched_at=now_iso,
                    reason_code="pdf-parse-error",
                    reason=f"pdf extract failed: {exc}",
                )
        elif "text/markdown" in ctype_low:
            source_via = "markdown-negotiated"
            enc = _detect_text_encoding(sresp.content, ctype, sresp.encoding)
            body = _normalize_extracted_text(sresp.content.decode(enc, errors="replace"))
            title = _extract_markdown_title(body)
            if _looks_like_site_root_llms_fallback(final_url_str, title, body):
                slug_tokens = _extract_path_slug_tokens(final_url_str)
                md_hits = _count_slug_hits(slug_tokens, f"{title} {body}")
                try:
                    hresp = await _fetch_stream_with_redirects(
                        cli,
                        final_url_str,
                        headers={
                            "User-Agent": ua,
                            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        },
                        timeout_sec=timeout_sec,
                        max_bytes=_MAX_BYTES,
                        resolve_host=resolve_host,
                        max_redirects=_MAX_REDIRECTS,
                    )
                    if hresp.status_code == 200 and hresp.content:
                        html_ctype = hresp.headers.get("content-type", "")
                        h_enc = _detect_text_encoding(hresp.content, html_ctype, hresp.encoding)
                        html_raw = hresp.content.decode(h_enc, errors="replace")
                        if "html" in html_ctype.lower() or "<html" in html_raw[:500].lower():
                            h_title, h_body = _extract_html_title_and_body(html_raw)
                            html_hits = _count_slug_hits(slug_tokens, f"{h_title} {h_body}")
                            if h_body and html_hits > md_hits:
                                source_via = "html-fallback-from-markdown"
                                title = h_title
                                body = h_body
                                ctype = html_ctype
                                resp_headers = hresp.headers
                                resp_status = hresp.status_code
                                size_capped = hresp.size_capped
                except Exception:
                    pass
        else:
            enc = _detect_text_encoding(sresp.content, ctype, sresp.encoding)
            text_raw = sresp.content.decode(enc, errors="replace")
            if "html" in ctype_low or "<html" in text_raw[:500].lower():
                title, body = _extract_html_title_and_body(text_raw)
            else:
                body = text_raw.strip()

        body = sanitize_text(body)
        if not body or (resp_status in (202, 204) and len(body) < 20):
            return FetchResult(
                ok=False,
                url=safe_url,
                final_url=redact_url_secrets(final_url_str),
                status_code=resp_status,
                content_type=ctype,
                source_via=source_via,
                title=title,
                fetched_at=now_iso,
                reason_code="empty-body",
                reason=f"HTTP {resp_status} returned empty/challenge body",
            )
        truncated = len(body) > max_chars or size_capped
        sig_map, md_toks = _parse_agentic_headers(resp_headers)
        return _finalize_and_store(
            FetchResult(
                ok=True,
                url=safe_url,
                final_url=redact_url_secrets(final_url_str),
                status_code=resp_status,
                content_type=ctype,
                source_via=source_via,
                title=title,
                content=body[:max_chars],
                truncated=truncated,
                fetched_at=now_iso,
                untrusted_content=True,
                content_signals=sig_map,
                markdown_tokens=md_toks,
            ),
            full_body=body,
        )
    except httpx.TimeoutException as exc:
        return FetchResult(ok=False, url=safe_url, fetched_at=now_iso, reason_code="deadline", reason=f"timeout: {exc}")
    except httpx.HTTPError as exc:
        return FetchResult(ok=False, url=safe_url, fetched_at=now_iso, reason_code="network-unavailable", reason=str(exc))
    finally:
        if own_client:
            await cli.aclose()


class BatchFetchResult(BaseModel):
    ok: bool
    total: int
    succeeded: int
    failed: int
    untrusted_content: bool = True
    items: list[FetchResult] = Field(default_factory=list)


async def safe_fetch_urls_batch(
    urls: list[str],
    max_chars: int = 8000,
    timeout_sec: float = 15.0,
    concurrency: int = 5,
    client: httpx.AsyncClient | None = None,
    resolve_host: bool = True,
    disk_cache: Any | None = None,
    use_cache: bool = True,
    prefer_llms_txt: bool = False,
    query: str = "",
) -> BatchFetchResult:
    """并发安全抓取多条 URL（自动去重、信号量限流、单条 SSRF/异常隔离）。"""
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        u = str(raw or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(u)
        if len(deduped) >= 20:
            break

    if not deduped:
        return BatchFetchResult(ok=False, total=0, succeeded=0, failed=0, untrusted_content=True, items=[])

    eff_conc = max(1, min(10, int(concurrency)))
    sem = asyncio.Semaphore(eff_conc)
    own_client = client is None
    cli = client or httpx.AsyncClient(follow_redirects=False)

    async def _one(u: str) -> FetchResult:
        async with sem:
            try:
                return await safe_fetch_url(
                    url=u,
                    max_chars=max_chars,
                    timeout_sec=timeout_sec,
                    client=cli,
                    resolve_host=resolve_host,
                    disk_cache=disk_cache,
                    use_cache=use_cache,
                    prefer_llms_txt=prefer_llms_txt,
                    query=query,
                )
            except Exception as exc:
                return FetchResult(
                    ok=False,
                    url=redact_url_secrets(u),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    reason_code="error",
                    reason=str(exc),
                )

    try:
        items = list(await asyncio.gather(*(_one(u) for u in deduped)))
    finally:
        if own_client:
            await cli.aclose()

    succ = sum(1 for it in items if it.ok)
    fail = len(items) - succ
    return BatchFetchResult(
        ok=succ > 0,
        total=len(items),
        succeeded=succ,
        failed=fail,
        untrusted_content=True,
        items=items,
    )
