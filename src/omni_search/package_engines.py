"""package_engines: 原生异步 PyPI 与 Go 包生态精准检索引擎（替代忽略 q 参数的 Ecosyste.ms /packages 假活端点）。"""

from __future__ import annotations

import asyncio
import html as _html
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from omni_search import __version__
from omni_search.contracts import (
    EngineEnvelope,
    EngineFailure,
    ReasonCode,
    RequestStats,
    SearchItem,
    redact_url_secrets,
    sanitize_text,
)

_PYPI_NOISE = frozenset({
    "python", "pypi", "pip", "package", "packages", "library", "libraries",
    "module", "modules", "sdk", "framework", "tool", "tools", "client", "server",
    "async", "http", "for", "with", "and", "the", "best", "top", "open", "source",
})
_PKG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_GO_SNIPPET_SPLIT_RE = re.compile(r'<div class="SearchSnippet"[\s\S]*?(?=<div class="SearchSnippet"|<div class="Pagination|$)')
_GO_HREF_RE = re.compile(
    r'<a[^>]*href="/([^"?#]+)"[^>]*data-gtmc="search result"|data-gtmc="search result"[^>]*href="/([^"?#]+)"',
    re.IGNORECASE,
)
_GO_DESC_RE = re.compile(r'class="SearchSnippet-synopsis"[^>]*>([\s\S]*?)</p>', re.IGNORECASE)
_GO_TAG_RE = re.compile(r"<[^>]+>")


def _extract_pypi_prefixes(query: str, max_terms: int = 2) -> list[str]:
    """从自然语言或包名查询中提炼高区分度 PyPI 前缀候选词。"""
    raw_toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}", query or "")
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for idx, tok in enumerate(raw_toks):
        low = tok.lower().strip("._-")
        if len(low) < 2 or low in _PYPI_NOISE or low in seen or low.isdigit():
            continue
        seen.add(low)
        score = 1.0
        if "-" in low or "_" in low:
            score += 2.5
        if any(c.isdigit() for c in low) and any(c.isalpha() for c in low):
            score += 1.5
        if len(low) >= 4:
            score += 0.5
        scored.append((score, idx, low))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in scored[:max_terms]]


class PyPiSearchEngine:
    """PyPI 精准包检索：先经 Ecosyste.ms `package_names?prefix=` 前缀索引查包名，再并发直取 PyPI 官方 JSON 元数据。"""

    id = "pypi_ecosystems"
    domain = "packages"
    tier = "L0"

    def __init__(self, deadline_sec: float = 12.0):
        self.deadline_sec = deadline_sec

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        eff_limit = max(1, min(int(limit), 12))
        prefixes = _extract_pypi_prefixes(query, max_terms=2)
        if not prefixes:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
            )

        headers = {
            "User-Agent": f"search-all-in-one/{__version__} (+https://github.com/omni-search)",
            "Accept": "application/json",
        }
        req_att = req_ok = req_fail = 0
        cand_names: list[str] = []
        seen_names: set[str] = set()

        def _add_name(n: str) -> None:
            clean = n.strip()
            low = clean.lower()
            if clean and low not in seen_names and _PKG_NAME_RE.fullmatch(low):
                seen_names.add(low)
                cand_names.append(clean)

        # 1) 前缀查询真实 PyPI 包名（避开忽略 q 的 /packages 端点）
        rate_limited = False
        net_err: str | None = None

        async def _probe_prefix(pfx: str) -> tuple[bool, list[str], int | None, str | None]:
            try:
                r = await client.get(
                    "https://packages.ecosyste.ms/api/v1/registries/pypi.org/package_names",
                    params={"prefix": pfx, "per_page": str(max(eff_limit, 6))},
                    headers=headers,
                    timeout=min(self.deadline_sec, 6.0),
                )
                if r.status_code == 200:
                    arr = r.json()
                    if isinstance(arr, list):
                        return True, [str(x) for x in arr if x], 200, None
                return False, [], r.status_code, f"HTTP {r.status_code}"
            except httpx.TimeoutException as exc:
                return False, [], None, f"timeout: {exc}"
            except Exception as exc:
                return False, [], None, str(exc)

        prefix_outs = await asyncio.gather(*(_probe_prefix(p) for p in prefixes))
        for pfx, (ok_flag, names, code, err_msg) in zip(prefixes, prefix_outs):
            req_att += 1
            if ok_flag:
                req_ok += 1
                if pfx in {n.lower() for n in names}:
                    _add_name(pfx)
                for n in names:
                    _add_name(n)
            else:
                req_fail += 1
                if code == 429:
                    rate_limited = True
                net_err = err_msg

        # 若前缀索引未命中或限流，对合法单包名直接尝试 PyPI 官方 JSON API
        if not cand_names:
            for pfx in prefixes:
                _add_name(pfx)

        target_pkgs = cand_names[:eff_limit]
        if not target_pkgs:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                failure=EngineFailure(
                    reason_code=ReasonCode.RATE_LIMITED if rate_limited else ReasonCode.ERROR,
                    reason=net_err or "pypi lookup failed",
                )
                if req_fail > 0 and req_ok == 0
                else None,
            )

        # 2) 并发直取 PyPI 官方 JSON 元数据
        async def _fetch_pypi(pkg: str) -> tuple[bool, SearchItem | None, bool]:
            try:
                r = await client.get(
                    f"https://pypi.org/pypi/{quote(pkg, safe='')}/json",
                    headers=headers,
                    timeout=min(self.deadline_sec, 6.0),
                )
                if r.status_code == 404:
                    return True, None, False
                if r.status_code != 200:
                    return False, None, r.status_code == 429
                data = r.json()
                info = data.get("info") if isinstance(data, dict) else None
                if not isinstance(info, dict):
                    return True, None, False
                name = sanitize_text(str(info.get("name") or pkg)).strip()
                ver = sanitize_text(str(info.get("version") or "")).strip()
                summary = sanitize_text(str(info.get("summary") or "")).strip()
                proj_url = redact_url_secrets(
                    str(info.get("project_url") or info.get("package_url") or f"https://pypi.org/project/{name}/")
                )
                title = f"{name} ({ver})" if ver else name
                return (
                    True,
                    SearchItem(
                        id=f"pypi:{name.lower()}",
                        domain=self.domain,
                        source=self.id,
                        title=title,
                        url=proj_url,
                        snippet=summary[:420] or f"PyPI package {name}",
                        modality="package",
                        extra={"package": name, "version": ver} if ver else {"package": name},
                    ),
                    False,
                )
            except Exception:
                return False, None, False

        meta_outs = await asyncio.gather(*(_fetch_pypi(p) for p in target_pkgs))
        results: list[SearchItem] = []
        for ok_flag, item, is_rl in meta_outs:
            req_att += 1
            if ok_flag:
                req_ok += 1
                if item is not None:
                    results.append(item)
            else:
                req_fail += 1
                if is_rl:
                    rate_limited = True

        failure: EngineFailure | None = None
        if not results and req_ok == 0 and req_fail > 0:
            failure = EngineFailure(
                reason_code=ReasonCode.RATE_LIMITED if rate_limited else ReasonCode.NETWORK_UNAVAILABLE,
                reason=net_err or "pypi metadata lookup failed",
            )
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=results[:eff_limit],
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
            failure=failure,
        )


def parse_pkg_go_dev_html(html_text: str, limit: int) -> list[SearchItem]:
    """解析 pkg.go.dev 官方搜索结果页 HTML，抽取模块路径与摘要。"""
    results: list[SearchItem] = []
    seen: set[str] = set()
    for block in _GO_SNIPPET_SPLIT_RE.findall(html_text or ""):
        if len(results) >= limit:
            break
        m_href = _GO_HREF_RE.search(block)
        if not m_href:
            continue
        mod_path = sanitize_text((m_href.group(1) or m_href.group(2) or "").strip().strip("/"))
        if not mod_path or mod_path.lower() in seen:
            continue
        seen.add(mod_path.lower())
        desc = ""
        m_desc = _GO_DESC_RE.search(block)
        if m_desc:
            raw_d = _html.unescape(_GO_TAG_RE.sub("", m_desc.group(1)))
            desc = sanitize_text(re.sub(r"\s+", " ", raw_d)).strip()
        results.append(
            SearchItem(
                id=f"go:{mod_path.lower()}",
                domain="packages",
                source="golang_ecosystems",
                title=mod_path,
                url=f"https://pkg.go.dev/{mod_path}",
                snippet=desc[:420] or f"Go module {mod_path}",
                modality="package",
                extra={"module": mod_path},
            )
        )
    return results


class GoPkgSearchEngine:
    """Go 官方包搜索引擎（pkg.go.dev/search，支持多词查询，免 Key L0）。"""

    id = "golang_ecosystems"
    domain = "packages"
    tier = "L0"

    def __init__(self, deadline_sec: float = 12.0):
        self.deadline_sec = deadline_sec

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        eff_limit = max(1, min(int(limit), 20))
        headers = {
            "User-Agent": f"search-all-in-one/{__version__} (+https://github.com/omni-search)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }
        req_stats = RequestStats(attempted=1)
        try:
            resp = await client.get(
                "https://pkg.go.dev/search",
                params={"q": query.strip(), "m": "package"},
                headers=headers,
                timeout=self.deadline_sec,
                follow_redirects=True,
            )
        except httpx.TimeoutException as exc:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason=f"timeout: {exc}"),
            )
        except httpx.HTTPError as exc:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.NETWORK_UNAVAILABLE, reason=str(exc)),
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code == 429:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.RATE_LIMITED, reason="HTTP 429 rate-limited"),
            )
        if resp.status_code >= 400:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"HTTP {resp.status_code}"),
            )

        req_stats.succeeded = 1
        items = parse_pkg_go_dev_html(resp.text, limit=eff_limit)
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=items,
            elapsed_ms=elapsed_ms,
            requests=req_stats,
        )
