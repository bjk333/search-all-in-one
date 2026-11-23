"""fetch_apis_extended: 学术/AI/社区/包生态扩展官方公开 API 直取模块（arXiv/crates.io/HF/Wikipedia/DOI/HN）。"""

from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import SplitResult, parse_qs, quote
import xml.etree.ElementTree as ET

import httpx

from omni_search import __version__
from omni_search.contracts import sanitize_text
from omni_search.html_markdown import _normalize_extracted_text, _unescape_html_entities

_MAX_BYTES = 2 * 1024 * 1024

_HF_RESERVED_PREFIXES = {
    "spaces", "papers", "docs", "blog", "models", "organizations", "settings",
    "login", "join", "pricing", "enterprise", "inference", "tasks", "collections",
}
_WIKI_HOST_RE = re.compile(r"^([a-z]{2,12})\.wikipedia\.org$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.([A-Za-z0-9._/-]{4,40})$", re.IGNORECASE)
_XML_TAG_RE = re.compile(r"<[^>]+>")


async def _fetch_arxiv_by_id(
    arxiv_id: str,
    safe_url: str,
    max_chars: int,
    timeout_sec: float,
    cli: httpx.AsyncClient,
    resolve_host: bool,
    now_iso: str,
    validate_url: Callable[..., tuple[bool, str | None]],
    result_cls: type[Any],
) -> Any | None:
    """通过 arXiv Atom API 拉取论文结构化元信息与摘要。"""
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
    if not ok_api:
        return None
    try:
        r = await cli.get(
            api_url,
            headers={"User-Agent": f"search-all-in-one/{__version__}", "Accept": "application/atom+xml"},
            timeout=timeout_sec,
            follow_redirects=False,
        )
        if r.status_code != 200 or not r.content:
            return None
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content[:_MAX_BYTES])
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        raw_title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        title_str = re.sub(r"\s+", " ", sanitize_text(raw_title)).strip()
        summary_raw = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        summary_str = re.sub(r"\s+", " ", sanitize_text(summary_raw)).strip()
        published = sanitize_text((entry.findtext("a:published", default="", namespaces=ns) or "").strip())[:10]
        authors = [
            sanitize_text((au.findtext("a:name", default="", namespaces=ns) or "").strip())
            for au in entry.findall("a:author", ns)
        ]
        authors = [a for a in authors if a]
        cats = [
            sanitize_text(str(c.attrib.get("term") or "").strip())
            for c in entry.findall("a:category", ns)
            if c.attrib.get("term")
        ]
        pdf_href = ""
        for lk in entry.findall("a:link", ns):
            if lk.attrib.get("title") == "pdf" or lk.attrib.get("type") == "application/pdf":
                pdf_href = sanitize_text(str(lk.attrib.get("href") or "").strip())
                break
        if not (title_str or summary_str):
            return None
        blocks = [f"# {title_str or arxiv_id}"]
        meta_parts: list[str] = [f"arXiv:{arxiv_id}"]
        if published:
            meta_parts.append(f"Published: {published}")
        if cats:
            meta_parts.append(f"Categories: {', '.join(cats[:6])}")
        blocks.append(" | ".join(meta_parts))
        if authors:
            blocks.append("Authors: " + ", ".join(authors[:12]))
        if pdf_href:
            blocks.append(f"PDF: {pdf_href}")
        if summary_str:
            blocks.append(f"## Abstract\n\n{summary_str}")
        body = _normalize_extracted_text("\n\n".join(blocks))
        if not body:
            return None
        return result_cls(
            ok=True,
            url=safe_url,
            final_url=api_url,
            status_code=200,
            content_type="application/atom+xml",
            source_via="arxiv-api",
            title=title_str or f"arXiv:{arxiv_id}",
            content=body[:max_chars],
            truncated=len(body) > max_chars,
            fetched_at=now_iso,
            untrusted_content=True,
        )
    except Exception:
        return None



async def _try_fetch_extended_official_api(
    url: str,
    safe_url: str,
    host: str,
    segs: list[str],
    parts: SplitResult,
    max_chars: int,
    timeout_sec: float,
    cli: httpx.AsyncClient,
    resolve_host: bool,
    now_iso: str,
    *,
    validate_url: Callable[..., tuple[bool, str | None]],
    result_cls: type[Any],
) -> Any | None:
    """处理 arXiv / crates.io / HuggingFace / Wikipedia / DOI / Hacker News 官方 API 直取。"""
    # 5) arxiv.org/abs/<id> -> https://export.arxiv.org/api/query?id_list=<id>
    if host == "arxiv.org" and len(segs) >= 2 and segs[0] == "abs":
        arxiv_id = "/".join(segs[1:]).strip()
        if arxiv_id and re.fullmatch(r"[A-Za-z0-9._/-]{4,40}", arxiv_id):
            arxiv_res = await _fetch_arxiv_by_id(
                arxiv_id, safe_url, max_chars, timeout_sec, cli, resolve_host, now_iso, validate_url, result_cls
            )
            if arxiv_res is not None:
                return arxiv_res

    # 6) crates.io/crates/<pkg> -> https://crates.io/api/v1/crates/<pkg>
    if host == "crates.io" and len(segs) >= 2 and segs[0] == "crates":
        crate_name = segs[1].strip()
        if crate_name and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", crate_name):
            api_url = f"https://crates.io/api/v1/crates/{crate_name}"
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if ok_api:
                try:
                    r = await cli.get(
                        api_url,
                        headers={"User-Agent": f"search-all-in-one/{__version__}", "Accept": "application/json"},
                        timeout=timeout_sec,
                        follow_redirects=False,
                    )
                    if r.status_code == 200:
                        crate = (r.json() or {}).get("crate") or {}
                        name = sanitize_text(str(crate.get("name") or crate_name)).strip()
                        ver = sanitize_text(str(crate.get("max_stable_version") or crate.get("newest_version") or "")).strip()
                        desc = sanitize_text(str(crate.get("description") or "")).strip()
                        repo = sanitize_text(str(crate.get("repository") or "")).strip()
                        docs = sanitize_text(str(crate.get("documentation") or "")).strip()
                        downloads = crate.get("downloads")
                        blocks = [f"# {name}" + (f" (v{ver})" if ver else "")]
                        if desc:
                            blocks.append(desc)
                        links: list[str] = []
                        if repo:
                            links.append(f"Repository: {repo}")
                        if docs:
                            links.append(f"Documentation: {docs}")
                        if isinstance(downloads, int) and downloads > 0:
                            links.append(f"Downloads: {downloads:,}")
                        if links:
                            blocks.append("\n".join(links))
                        body = _normalize_extracted_text("\n\n".join(blocks))
                        if body:
                            return result_cls(
                                ok=True,
                                url=safe_url,
                                final_url=api_url,
                                status_code=200,
                                content_type="application/json",
                                source_via="crates-io-api",
                                title=f"{name}" + (f" v{ver}" if ver else ""),
                                content=body[:max_chars],
                                truncated=len(body) > max_chars,
                                fetched_at=now_iso,
                                untrusted_content=True,
                            )
                except Exception:
                    pass

    # 7) huggingface.co/<owner>/<model> 或 /datasets/<owner>/<ds> -> HF Hub API
    if host == "huggingface.co" and len(segs) >= 2:
        hf_kind: str | None = None
        hf_repo_id = ""
        if segs[0].lower() == "datasets" and len(segs) == 3:
            hf_kind = "datasets"
            hf_repo_id = f"{segs[1]}/{segs[2]}"
        elif len(segs) == 2 and segs[0].lower() not in _HF_RESERVED_PREFIXES:
            hf_kind = "models"
            hf_repo_id = f"{segs[0]}/{segs[1]}"
        if hf_kind and hf_repo_id:
            api_url = f"https://huggingface.co/api/{hf_kind}/{hf_repo_id}"
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if ok_api:
                try:
                    r = await cli.get(
                        api_url,
                        headers={"User-Agent": "search-all-in-one/0.9", "Accept": "application/json"},
                        timeout=timeout_sec,
                        follow_redirects=False,
                    )
                    if r.status_code == 200:
                        data = r.json() or {}
                        rid = sanitize_text(str(data.get("id") or hf_repo_id)).strip()
                        pipe = sanitize_text(str(data.get("pipeline_tag") or "")).strip()
                        lib = sanitize_text(str(data.get("library_name") or "")).strip()
                        desc = sanitize_text(str(data.get("description") or "")).strip()
                        downloads = data.get("downloads")
                        likes = data.get("likes")
                        tags = [sanitize_text(str(t)).strip() for t in (data.get("tags") or [])[:12] if str(t).strip()]
                        card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
                        lic = sanitize_text(str(card.get("license") or "")).strip()
                        blocks = [f"# {rid} ({hf_kind[:-1]})"]
                        meta_parts: list[str] = []
                        if pipe:
                            meta_parts.append(f"Task: {pipe}")
                        if lib:
                            meta_parts.append(f"Library: {lib}")
                        if lic:
                            meta_parts.append(f"License: {lic}")
                        if isinstance(downloads, int):
                            meta_parts.append(f"Downloads: {downloads:,}")
                        if isinstance(likes, int):
                            meta_parts.append(f"Likes: {likes:,}")
                        if meta_parts:
                            blocks.append(" | ".join(meta_parts))
                        if desc:
                            blocks.append(desc)
                        if tags:
                            blocks.append("Tags: " + ", ".join(tags))
                        body = _normalize_extracted_text("\n\n".join(blocks))
                        if body:
                            return result_cls(
                                ok=True,
                                url=safe_url,
                                final_url=api_url,
                                status_code=200,
                                content_type="application/json",
                                source_via="huggingface-api",
                                title=rid,
                                content=body[:max_chars],
                                truncated=len(body) > max_chars,
                                fetched_at=now_iso,
                                untrusted_content=True,
                            )
                except Exception:
                    pass

    # 8) <lang>.wikipedia.org/wiki/<title> -> Wikimedia REST v1 Summary API
    m_wiki = _WIKI_HOST_RE.match(host)
    if m_wiki and len(segs) == 2 and segs[0] == "wiki" and ":" not in segs[1]:
        lang = m_wiki.group(1)
        page_slug = segs[1].strip()
        if page_slug:
            api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page_slug}"
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if ok_api:
                try:
                    r = await cli.get(
                        api_url,
                        headers={"User-Agent": "search-all-in-one/0.9", "Accept": "application/json"},
                        timeout=timeout_sec,
                        follow_redirects=False,
                    )
                    if r.status_code == 200:
                        data = r.json() or {}
                        wtitle = sanitize_text(str(data.get("title") or page_slug)).strip()
                        wdesc = sanitize_text(str(data.get("description") or "")).strip()
                        extract = sanitize_text(str(data.get("extract") or "")).strip()
                        page_url = sanitize_text(
                            str(((data.get("content_urls") or {}).get("desktop") or {}).get("page") or "")
                        ).strip()
                        if extract:
                            blocks = [f"# {wtitle}"]
                            if wdesc:
                                blocks.append(wdesc)
                            blocks.append(extract)
                            if page_url:
                                blocks.append(f"Source: {page_url}")
                            body = _normalize_extracted_text("\n\n".join(blocks))
                            if body:
                                return result_cls(
                                    ok=True,
                                    url=safe_url,
                                    final_url=api_url,
                                    status_code=200,
                                    content_type="application/json",
                                    source_via="wikipedia-rest-api",
                                    title=wtitle,
                                    content=body[:max_chars],
                                    truncated=len(body) > max_chars,
                                    fetched_at=now_iso,
                                    untrusted_content=True,
                                )
                except Exception:
                    pass

    # 9) doi.org/<doi> 或 dx.doi.org/<doi> -> arXiv DOI 直连 Atom API / 通用 DOI 走 Crossref Works API
    if host in ("doi.org", "dx.doi.org") and len(segs) >= 2:
        doi = "/".join(segs).strip()
        m_arxiv_doi = _ARXIV_DOI_RE.match(doi)
        if m_arxiv_doi:
            arxiv_res = await _fetch_arxiv_by_id(
                m_arxiv_doi.group(1), safe_url, max_chars, timeout_sec, cli, resolve_host, now_iso, validate_url, result_cls
            )
            if arxiv_res is not None:
                return arxiv_res
        if _DOI_RE.match(doi):
            api_url = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if ok_api:
                try:
                    r = await cli.get(
                        api_url,
                        headers={"User-Agent": f"search-all-in-one/{__version__} (mailto:omni-search@localhost)", "Accept": "application/json"},
                        timeout=timeout_sec,
                        follow_redirects=False,
                    )
                    if r.status_code == 200:
                        msg = (r.json() or {}).get("message") or {}
                        titles = msg.get("title") or []
                        title_str = sanitize_text(str(titles[0] if isinstance(titles, list) and titles else doi)).strip()
                        containers = msg.get("container-title") or []
                        venue = sanitize_text(str(containers[0] if isinstance(containers, list) and containers else "")).strip()
                        year = ""
                        for date_key in ("published-print", "published-online", "issued", "created"):
                            dparts = (msg.get(date_key) or {}).get("date-parts") or []
                            if dparts and isinstance(dparts[0], list) and dparts[0]:
                                year = str(dparts[0][0])
                                break
                        cites = msg.get("is-referenced-by-count")
                        authors: list[str] = []
                        for au in (msg.get("author") or [])[:12]:
                            if isinstance(au, dict):
                                full = " ".join(p for p in (str(au.get("given") or "").strip(), str(au.get("family") or "").strip()) if p)
                                if full:
                                    authors.append(sanitize_text(full))
                        pub_url = sanitize_text(str(msg.get("URL") or "")).strip()
                        raw_abs = str(msg.get("abstract") or "").strip()
                        clean_abs = re.sub(r"\s+", " ", sanitize_text(_unescape_html_entities(_XML_TAG_RE.sub(" ", raw_abs)))).strip()
                        blocks = [f"# {title_str}"]
                        meta_parts = [f"DOI: {doi}"]
                        if year:
                            meta_parts.append(f"Year: {year}")
                        if venue:
                            meta_parts.append(f"Venue: {venue}")
                        if isinstance(cites, int):
                            meta_parts.append(f"Citations: {cites:,}")
                        blocks.append(" | ".join(meta_parts))
                        if authors:
                            blocks.append("Authors: " + ", ".join(authors))
                        if pub_url:
                            blocks.append(f"URL: {pub_url}")
                        if clean_abs:
                            blocks.append(f"## Abstract\n\n{clean_abs}")
                        body = _normalize_extracted_text("\n\n".join(blocks))
                        if body:
                            return result_cls(
                                ok=True,
                                url=safe_url,
                                final_url=api_url,
                                status_code=200,
                                content_type="application/json",
                                source_via="crossref-doi-api",
                                title=title_str,
                                content=body[:max_chars],
                                truncated=len(body) > max_chars,
                                fetched_at=now_iso,
                                untrusted_content=True,
                            )
                except Exception:
                    pass

    # 10) news.ycombinator.com/item?id=<id> -> Hacker News Official Firebase API
    if host == "news.ycombinator.com" and len(segs) == 1 and segs[0] == "item":
        qs = parse_qs(parts.query)
        item_ids = qs.get("id") or []
        item_id = item_ids[0].strip() if item_ids else ""
        if item_id.isdigit():
            api_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if ok_api:
                try:
                    r = await cli.get(
                        api_url,
                        headers={"User-Agent": f"search-all-in-one/{__version__}", "Accept": "application/json"},
                        timeout=timeout_sec,
                        follow_redirects=False,
                    )
                    if r.status_code == 200:
                        data = r.json() or {}
                        if isinstance(data, dict) and data.get("id"):
                            htitle = sanitize_text(str(data.get("title") or f"HN #{item_id}")).strip()
                            htype = sanitize_text(str(data.get("type") or "story")).strip()
                            author = sanitize_text(str(data.get("by") or "")).strip()
                            score = data.get("score")
                            descendants = data.get("descendants")
                            ext_url = sanitize_text(str(data.get("url") or "")).strip()
                            raw_text = str(data.get("text") or "").strip()
                            clean_text = _normalize_extracted_text(
                                sanitize_text(_unescape_html_entities(_XML_TAG_RE.sub("\n", raw_text)))
                            ) if raw_text else ""
                            blocks = [f"# {htitle}"]
                            meta_parts = [f"Type: {htype}"]
                            if author:
                                meta_parts.append(f"By: {author}")
                            if isinstance(score, int):
                                meta_parts.append(f"Score: {score}")
                            if isinstance(descendants, int):
                                meta_parts.append(f"Comments: {descendants}")
                            blocks.append(" | ".join(meta_parts))
                            if ext_url:
                                blocks.append(f"URL: {ext_url}")
                            if clean_text:
                                blocks.append(clean_text)
                            body = _normalize_extracted_text("\n\n".join(blocks))
                            if body:
                                return result_cls(
                                    ok=True,
                                    url=safe_url,
                                    final_url=api_url,
                                    status_code=200,
                                    content_type="application/json",
                                    source_via="hackernews-api",
                                    title=htitle,
                                    content=body[:max_chars],
                                    truncated=len(body) > max_chars,
                                    fetched_at=now_iso,
                                    untrusted_content=True,
                                )
                except Exception:
                    pass

    return None
