"""site_map: 带 SSRF 防护的受控站点目录映射（/llms.txt -> /sitemap.xml -> 同源 HTML 链接）。"""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from omni_search.contracts import redact_url_secrets, sanitize_text
from omni_search.fetcher import FetchResult, _validate_target_url

_MAX_MAP_BYTES = 1024 * 1024
_STATIC_EXTS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".css", ".js", ".mjs", ".map", ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".zip", ".tar", ".gz",
)
_MD_LINK_RE = re.compile(r"\[([^\]]{1,140})\]\((https?://[^\s)]+|/[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"(?<!\()(https?://[^\s<>)\"']+)")


class SiteMapEntry(BaseModel):
    url: str
    title: str = ""
    source: str = "html-links"
    score: float = 0.0


class SiteMapResult(BaseModel):
    ok: bool
    url: str
    host: str = ""
    sources_used: list[str] = Field(default_factory=list)
    total_found: int = 0
    urls: list[SiteMapEntry] = Field(default_factory=list)
    fetched_at: str = ""
    untrusted_content: bool = True
    reason_code: str | None = None
    reason: str | None = None


class _AnchorExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self._buf: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self._href = v.strip()
                    self._buf = []
                    break

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            txt = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            self.links.append((self._href, txt[:140]))
            self._href = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._buf.append(data)


def _norm_host(host: str) -> str:
    h = (host or "").strip().lower()
    return h[4:] if h.startswith("www.") else h


def _clean_same_host_url(raw_href: str, base_url: str, target_host: str) -> str | None:
    if not raw_href or raw_href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
        return None
    full = urljoin(base_url, raw_href.strip())
    p = urlsplit(full)
    if p.scheme.lower() not in ("http", "https"):
        return None
    if _norm_host(p.hostname or "") != target_host:
        return None
    path_low = (p.path or "").lower()
    if any(path_low.endswith(ext) for ext in _STATIC_EXTS):
        return None
    clean_path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), clean_path, "", ""))


def _score_entry(entry: SiteMapEntry, qterms: list[str]) -> float:
    is_llms = entry.source in ("llms-txt", "llms-full-txt")
    if not qterms:
        # 无 query 时优先 llms-txt/llms-full-txt 与较浅文档路径
        depth = max(1, len([s for s in urlsplit(entry.url).path.split("/") if s]))
        src_bonus = 2.0 if is_llms else (1.0 if entry.source == "sitemap" else 0.5)
        return round(src_bonus + 1.0 / depth, 4)
    hay_url = entry.url.lower()
    hay_title = entry.title.lower()
    hits = 0.0
    for t in qterms:
        if t in hay_title:
            hits += 2.0
        if t in hay_url:
            hits += 1.5
    src_bonus = 0.25 if is_llms else 0.0
    return round(hits / len(qterms) + src_bonus, 4)


def _extract_query_terms(query: str) -> list[str]:
    if not query.strip():
        return []
    toks = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", query.lower())
    return list(dict.fromkeys(toks))[:12]


def _parse_sitemap_locs(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """解析 sitemap XML，返回 (普通页面 loc 列表, 子 sitemap loc 列表)。"""
    pages: list[str] = []
    sub_maps: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return [], []
    root_tag = root.tag.lower()
    is_index = root_tag.endswith("sitemapindex")
    for elem in root.iter():
        if elem.tag.lower().endswith("loc") and elem.text and elem.text.strip():
            loc = elem.text.strip()
            if is_index:
                sub_maps.append(loc)
            else:
                pages.append(loc)
    return pages, sub_maps


async def safe_map_site(
    url: str,
    query: str = "",
    max_urls: int = 50,
    timeout_sec: float = 12.0,
    client: httpx.AsyncClient | None = None,
    resolve_host: bool = True,
) -> SiteMapResult:
    """受控站点目录映射：逐层探测 /llms.txt -> /sitemap.xml -> 同源 HTML 链接，全程强制 SSRF 校验。"""
    now_iso = datetime.now(timezone.utc).isoformat()
    safe_url = redact_url_secrets(url)
    allowed, block_reason = _validate_target_url(url, resolve_host=resolve_host)
    if not allowed:
        return SiteMapResult(
            ok=False,
            url=safe_url,
            fetched_at=now_iso,
            untrusted_content=True,
            reason_code="ssrf-blocked",
            reason=block_reason,
        )

    parts = urlsplit(url.strip())
    origin = f"{parts.scheme.lower()}://{parts.netloc}"
    target_host = _norm_host(parts.hostname or "")
    headers = {"User-Agent": "search-all-in-one/0.3 (+https://github.com/omni-search)", "Accept": "text/plain, application/xml, text/html"}

    entries_by_url: dict[str, SiteMapEntry] = {}
    sources_used: list[str] = []

    def _add(raw_u: str, title: str, src: str) -> None:
        cleaned = _clean_same_host_url(raw_u, origin, target_host)
        if not cleaned:
            return
        clean_title = sanitize_text(title).strip()
        if cleaned in entries_by_url:
            if not entries_by_url[cleaned].title and clean_title:
                entries_by_url[cleaned].title = clean_title
            return
        entries_by_url[cleaned] = SiteMapEntry(url=cleaned, title=clean_title, source=src)

    own_client = client is None
    cli = client or httpx.AsyncClient(follow_redirects=False)
    try:
        # 1) /llms.txt（若无命中则自动探测 /llms-full.txt）
        for suffix, src_tag in (("/llms.txt", "llms-txt"), ("/llms-full.txt", "llms-full-txt")):
            if sources_used:
                break
            llms_url = f"{origin}{suffix}"
            if _validate_target_url(llms_url, resolve_host=resolve_host)[0]:
                try:
                    r = await cli.get(llms_url, headers=headers, timeout=min(timeout_sec, 6.0), follow_redirects=False)
                    if r.status_code == 200 and "html" not in r.headers.get("content-type", "").lower():
                        txt = r.content[:_MAX_MAP_BYTES].decode("utf-8", errors="replace")
                        if "<html" not in txt[:300].lower():
                            before = len(entries_by_url)
                            for m in _MD_LINK_RE.finditer(txt):
                                _add(m.group(2), m.group(1), src_tag)
                            for m in _BARE_URL_RE.finditer(txt):
                                _add(m.group(1).rstrip(".,;)"), "", src_tag)
                            if len(entries_by_url) > before:
                                sources_used.append(src_tag)
                except Exception:
                    pass

        # 2) /sitemap.xml（支持最多展开 2 个子 sitemap）
        sm_url = f"{origin}/sitemap.xml"
        if _validate_target_url(sm_url, resolve_host=resolve_host)[0]:
            try:
                r = await cli.get(sm_url, headers=headers, timeout=min(timeout_sec, 6.0), follow_redirects=False)
                if r.status_code == 200:
                    pages, sub_maps = _parse_sitemap_locs(r.content[:_MAX_MAP_BYTES])
                    before = len(entries_by_url)
                    for loc in pages[:200]:
                        _add(loc, "", "sitemap")
                    for sub in sub_maps[:2]:
                        if _validate_target_url(sub, resolve_host=resolve_host)[0] and _norm_host(urlsplit(sub).hostname or "") == target_host:
                            try:
                                sr = await cli.get(sub, headers=headers, timeout=min(timeout_sec, 5.0), follow_redirects=False)
                                if sr.status_code == 200:
                                    spages, _ = _parse_sitemap_locs(sr.content[:_MAX_MAP_BYTES])
                                    for loc in spages[:150]:
                                        _add(loc, "", "sitemap")
                            except Exception:
                                pass
                    if len(entries_by_url) > before:
                        sources_used.append("sitemap")
            except Exception:
                pass

        # 3) 若上述不足 5 条，回退抓取目标页 HTML 抽取同源导航链接
        if len(entries_by_url) < 5:
            try:
                r = await cli.get(url, headers=headers, timeout=timeout_sec, follow_redirects=False)
                if r.status_code == 200:
                    raw_html = r.content[:_MAX_MAP_BYTES].decode(r.encoding or "utf-8", errors="replace")
                    parser = _AnchorExtractor()
                    parser.feed(raw_html)
                    before = len(entries_by_url)
                    for href, txt in parser.links[:300]:
                        _add(href, txt, "html-links")
                    if len(entries_by_url) > before:
                        sources_used.append("html-links")
            except Exception:
                pass

        qterms = _extract_query_terms(query)
        all_entries = list(entries_by_url.values())
        for e in all_entries:
            e.score = _score_entry(e, qterms)
        all_entries.sort(key=lambda e: (e.score, e.url), reverse=True)
        capped = all_entries[: max(1, min(200, int(max_urls)))]
        return SiteMapResult(
            ok=len(capped) > 0,
            url=safe_url,
            host=target_host,
            sources_used=sources_used,
            total_found=len(all_entries),
            urls=capped,
            fetched_at=now_iso,
            untrusted_content=True,
            reason_code=None if capped else "empty-map",
            reason=None if capped else "no same-host URLs discovered via llms.txt, sitemap.xml, or page links",
        )
    except httpx.TimeoutException as exc:
        return SiteMapResult(ok=False, url=safe_url, host=target_host, fetched_at=now_iso, reason_code="deadline", reason=str(exc))
    except httpx.HTTPError as exc:
        return SiteMapResult(ok=False, url=safe_url, host=target_host, fetched_at=now_iso, reason_code="network-unavailable", reason=str(exc))
    finally:
        if own_client:
            await cli.aclose()


class SiteCrawlResult(BaseModel):
    ok: bool
    url: str
    query: str = ""
    mapped_total: int = 0
    crawled_total: int = 0
    succeeded: int = 0
    failed: int = 0
    sources_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    untrusted_content: bool = True
    pages: list[FetchResult] = Field(default_factory=list)
    reason_code: str | None = None
    reason: str | None = None


async def safe_crawl_site(
    url: str,
    query: str = "",
    max_pages: int = 6,
    max_chars_per_page: int = 4000,
    concurrency: int = 4,
    timeout_sec: float = 15.0,
    client: httpx.AsyncClient | None = None,
    resolve_host: bool = True,
    disk_cache: Any | None = None,
    use_cache: bool = True,
) -> SiteCrawlResult:
    """站内定向采集：先 safe_map_site 发现并按 query 相关性排序，再 safe_fetch_urls_batch 并发抓取 Top-N 正文。"""
    from omni_search.fetcher import safe_fetch_urls_batch

    eff_pages = max(1, min(15, int(max_pages)))
    mres = await safe_map_site(
        url=url,
        query=query,
        max_urls=max(eff_pages * 3, 20),
        timeout_sec=timeout_sec,
        client=client,
        resolve_host=resolve_host,
    )
    if mres.reason_code == "ssrf-blocked":
        return SiteCrawlResult(
            ok=False,
            url=mres.url,
            query=query,
            untrusted_content=True,
            reason_code="ssrf-blocked",
            reason=mres.reason,
        )

    target_urls = [e.url for e in mres.urls[:eff_pages]]
    if not target_urls:
        target_urls = [url]

    bres = await safe_fetch_urls_batch(
        urls=target_urls,
        max_chars=max_chars_per_page,
        timeout_sec=timeout_sec,
        concurrency=concurrency,
        client=client,
        resolve_host=resolve_host,
        disk_cache=disk_cache,
        use_cache=use_cache,
        query=query,
    )
    crawl_warnings: list[str] = []
    for pg in bres.items:
        sigs = getattr(pg, "content_signals", None) or {}
        restricted = [f"{k}={sigs[k]}" for k in ("ai-input", "search") if sigs.get(k) == "no"]
        if restricted:
            crawl_warnings.append(f"content-signal-restricted: {pg.url} ({', '.join(restricted)})")
    return SiteCrawlResult(
        ok=bres.ok,
        url=mres.url,
        query=query,
        mapped_total=mres.total_found,
        crawled_total=bres.total,
        succeeded=bres.succeeded,
        failed=bres.failed,
        sources_used=mres.sources_used,
        warnings=crawl_warnings,
        untrusted_content=True,
        pages=bres.items,
        reason_code=None if bres.ok else (mres.reason_code or "crawl-failed"),
        reason=None if bres.ok else (mres.reason or "all crawled pages failed"),
    )
