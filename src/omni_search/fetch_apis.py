"""fetch_apis: 高频技术/学术/社区平台官方公开 API 直取模块（零密钥优先 + 全程 SSRF 校验）。"""

from __future__ import annotations

import os
import re
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import httpx

from omni_search import __version__
from omni_search.contracts import resolve_github_token, sanitize_text
from omni_search.fetch_apis_extended import (  # noqa: F401 - re-exported for compatibility
    _ARXIV_DOI_RE,
    _DOI_RE,
    _HF_RESERVED_PREFIXES,
    _WIKI_HOST_RE,
    _XML_TAG_RE,
    _fetch_arxiv_by_id,
    _try_fetch_extended_official_api,
)
from omni_search.html_markdown import _normalize_extracted_text

_MAX_BYTES = 2 * 1024 * 1024  # 普通网页/文本 2 MiB 硬上限

_GITHUB_RESERVED_OWNERS = {
    "topics", "search", "explore", "orgs", "settings", "features", "pricing",
    "marketplace", "sponsors", "login", "signup", "about", "security", "enterprise",
}


async def _try_fetch_official_api(
    url: str,
    safe_url: str,
    max_chars: int,
    timeout_sec: float,
    cli: httpx.AsyncClient,
    resolve_host: bool,
    now_iso: str,
    *,
    validate_url: Callable[..., tuple[bool, str | None]] | None = None,
    result_cls: type[Any] | None = None,
) -> Any | None:
    """对 9 大高频平台（npm/PyPI/GitHub/arXiv/crates.io/HF/Wikipedia/DOI/HackerNews）优先走官方公开 API 获取干净正文（失败平滑回退）。"""
    if validate_url is None or result_cls is None:
        from omni_search.fetcher import FetchResult, _validate_target_url

        validate_url = validate_url or _validate_target_url
        result_cls = result_cls or FetchResult

    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    segs = [s for s in parts.path.split("/") if s]

    # 1) npmjs.com/package/<pkg> -> 优先 /<pkg>/latest（避免拉取 10-50MB 全量历史 packument），失败回退 /<pkg>
    if host == "npmjs.com" and len(segs) >= 2 and segs[0] == "package":
        if segs[1].startswith("@") and len(segs) >= 3:
            pkg = f"{segs[1]}/{segs[2]}"
        else:
            pkg = segs[1]
        latest_url = f"https://registry.npmjs.org/{pkg}/latest"
        full_url = f"https://registry.npmjs.org/{pkg}"
        for api_url in (latest_url, full_url):
            ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
            if not ok_api:
                continue
            try:
                r = await cli.get(
                    api_url,
                    headers={"User-Agent": f"search-all-in-one/{__version__}", "Accept": "application/json"},
                    timeout=timeout_sec,
                    follow_redirects=False,
                )
                if r.status_code != 200 or len(r.content) > _MAX_BYTES:
                    continue
                data = r.json()
                if not isinstance(data, dict):
                    continue
                name = sanitize_text(str(data.get("name") or pkg)).strip()
                latest = sanitize_text(
                    str(data.get("version") or (data.get("dist-tags") or {}).get("latest") or "")
                ).strip()
                desc = sanitize_text(str(data.get("description") or "")).strip()
                readme = sanitize_text(str(data.get("readme") or "")).strip()
                homepage = sanitize_text(str(data.get("homepage") or "")).strip()
                header = f"# {name}" + (f" (v{latest})" if latest else "")
                blocks = [header]
                if desc:
                    blocks.append(desc)
                if homepage and homepage.startswith(("http://", "https://")):
                    blocks.append(f"Homepage: {homepage}")
                if readme and readme.lower() != "error: no readme data found":
                    blocks.append(readme)
                body = _normalize_extracted_text("\n\n".join(blocks))
                if not body:
                    continue
                return result_cls(
                    ok=True,
                    url=safe_url,
                    final_url=api_url,
                    status_code=200,
                    content_type="application/json",
                    source_via="npm-registry-api",
                    title=f"{name}" + (f" v{latest}" if latest else ""),
                    content=body[:max_chars],
                    truncated=len(body) > max_chars,
                    fetched_at=now_iso,
                    untrusted_content=True,
                )
            except Exception:
                continue
        return None

    # 1b) pypi.org/project/<pkg> -> https://pypi.org/pypi/<pkg>/json
    if host == "pypi.org" and len(segs) >= 2 and segs[0] == "project":
        pkg = segs[1]
        api_url = f"https://pypi.org/pypi/{pkg}/json"
        ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
        if ok_api:
            try:
                r = await cli.get(
                    api_url,
                    headers={"User-Agent": "search-all-in-one/0.3", "Accept": "application/json"},
                    timeout=timeout_sec,
                    follow_redirects=False,
                )
                if r.status_code == 200:
                    info = (r.json() or {}).get("info") or {}
                    name = sanitize_text(str(info.get("name") or pkg)).strip()
                    ver = sanitize_text(str(info.get("version") or "")).strip()
                    summary = sanitize_text(str(info.get("summary") or "")).strip()
                    desc = sanitize_text(str(info.get("description") or "")).strip()
                    header = f"# {name}" + (f" (v{ver})" if ver else "")
                    blocks = [header]
                    if summary:
                        blocks.append(summary)
                    if desc:
                        blocks.append(desc)
                    body = _normalize_extracted_text("\n\n".join(blocks))
                    if body:
                        return result_cls(
                            ok=True,
                            url=safe_url,
                            final_url=api_url,
                            status_code=200,
                            content_type="application/json",
                            source_via="pypi-json-api",
                            title=f"{name}" + (f" v{ver}" if ver else ""),
                            content=body[:max_chars],
                            truncated=len(body) > max_chars,
                            fetched_at=now_iso,
                            untrusted_content=True,
                        )
            except Exception:
                pass

    # 2) github.com/<owner>/<repo> -> https://api.github.com/repos/<owner>/<repo>/readme
    if host == "github.com" and len(segs) == 2 and segs[0].lower() not in _GITHUB_RESERVED_OWNERS:
        owner = segs[0]
        repo = segs[1][:-4] if segs[1].endswith(".git") else segs[1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
        if not ok_api:
            return None
        gh_headers = {
            "User-Agent": "omni-search/0.2",
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        gh_token = resolve_github_token()
        if gh_token:
            gh_headers["Authorization"] = f"Bearer {gh_token}"
        try:
            r = await cli.get(api_url, headers=gh_headers, timeout=timeout_sec, follow_redirects=False)
            if r.status_code != 200:
                return None
            raw_md = r.content[:_MAX_BYTES].decode("utf-8", errors="replace")
            body = sanitize_text(_normalize_extracted_text(raw_md))
            if not body:
                return None
            return result_cls(
                ok=True,
                url=safe_url,
                final_url=api_url,
                status_code=200,
                content_type=r.headers.get("content-type", "text/markdown"),
                source_via="github-readme-api",
                title=f"{owner}/{repo}",
                content=body[:max_chars],
                truncated=len(body) > max_chars,
                fetched_at=now_iso,
                untrusted_content=True,
            )
        except Exception:
            return None

    # 3) github.com/<owner>/<repo>/issues/<num> 或 /pull/<num> -> 官方 Issues API
    if (
        host == "github.com"
        and len(segs) == 4
        and segs[0].lower() not in _GITHUB_RESERVED_OWNERS
        and segs[2] in ("issues", "pull")
        and segs[3].isdigit()
    ):
        owner, repo, num = segs[0], segs[1], segs[3]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
        ok_api, _ = validate_url(api_url, resolve_host=resolve_host)
        if ok_api:
            gh_headers = {
                "User-Agent": "search-all-in-one/0.2",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            gh_token = resolve_github_token()
            if gh_token:
                gh_headers["Authorization"] = f"Bearer {gh_token}"
            try:
                r = await cli.get(api_url, headers=gh_headers, timeout=timeout_sec, follow_redirects=False)
                if r.status_code == 200:
                    data = r.json()
                    title_str = sanitize_text(str(data.get("title") or f"{owner}/{repo}#{num}")).strip()
                    state = sanitize_text(str(data.get("state") or "")).strip()
                    labels = [
                        sanitize_text(str(lb.get("name") or "")).strip()
                        for lb in (data.get("labels") or [])
                        if isinstance(lb, dict) and lb.get("name")
                    ]
                    body_md = sanitize_text(str(data.get("body") or "")).strip()
                    meta_line = f"State: {state}" + (f" | Labels: {', '.join(labels)}" if labels else "")
                    body = _normalize_extracted_text(f"# {title_str}\n\n{meta_line}\n\n{body_md}")
                    if body:
                        return result_cls(
                            ok=True,
                            url=safe_url,
                            final_url=api_url,
                            status_code=200,
                            content_type="application/json",
                            source_via="github-issue-api",
                            title=f"{owner}/{repo}#{num}: {title_str}",
                            content=body[:max_chars],
                            truncated=len(body) > max_chars,
                            fetched_at=now_iso,
                            untrusted_content=True,
                        )
            except Exception:
                pass

    # 4) github.com/<owner>/<repo>/blob/<ref>/<path...> -> raw.githubusercontent.com
    if (
        host == "github.com"
        and len(segs) >= 5
        and segs[0].lower() not in _GITHUB_RESERVED_OWNERS
        and segs[2] == "blob"
    ):
        raw_url = f"https://raw.githubusercontent.com/{segs[0]}/{segs[1]}/{segs[3]}/{'/'.join(segs[4:])}"
        ok_raw, _ = validate_url(raw_url, resolve_host=resolve_host)
        if ok_raw:
            try:
                r = await cli.get(
                    raw_url,
                    headers={"User-Agent": "search-all-in-one/0.2", "Accept": "text/plain"},
                    timeout=timeout_sec,
                    follow_redirects=False,
                )
                if r.status_code == 200:
                    raw_txt = r.content[:_MAX_BYTES].decode("utf-8", errors="replace")
                    body = sanitize_text(_normalize_extracted_text(raw_txt))
                    if body:
                        return result_cls(
                            ok=True,
                            url=safe_url,
                            final_url=raw_url,
                            status_code=200,
                            content_type=r.headers.get("content-type", "text/plain"),
                            source_via="github-raw-blob",
                            title=f"{segs[0]}/{segs[1]}/{'/'.join(segs[4:])}",
                            content=body[:max_chars],
                            truncated=len(body) > max_chars,
                            fetched_at=now_iso,
                            untrusted_content=True,
                        )
            except Exception:
                pass


    return await _try_fetch_extended_official_api(
        url,
        safe_url,
        host,
        segs,
        parts,
        max_chars,
        timeout_sec,
        cli,
        resolve_host,
        now_iso,
        validate_url=validate_url,
        result_cls=result_cls,
    )
