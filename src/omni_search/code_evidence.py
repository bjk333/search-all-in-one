"""原生跨仓公网代码片段证据引擎（grep.app 官方 MCP SSE + Sourcegraph SSE + GitHub Code Search API）。"""

from __future__ import annotations

import json
import os
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

_GREP_FILTER_RE = re.compile(r"(?:^|\s)(lang|repo|path):([^\s]+)", re.IGNORECASE)
_GREP_LANG_ALIAS: dict[str, str] = {
    "py": "Python",
    "python": "Python",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "tsx": "TSX",
    "jsx": "JSX",
    "rs": "Rust",
    "rust": "Rust",
    "go": "Go",
    "golang": "Go",
    "java": "Java",
    "kt": "Kotlin",
    "kotlin": "Kotlin",
    "c": "C",
    "cpp": "C++",
    "c++": "C++",
    "cs": "C#",
    "csharp": "C#",
    "rb": "Ruby",
    "ruby": "Ruby",
    "php": "PHP",
    "sh": "Shell",
    "bash": "Shell",
    "shell": "Shell",
    "ps1": "PowerShell",
    "powershell": "PowerShell",
    "md": "Markdown",
    "markdown": "Markdown",
    "yaml": "YAML",
    "yml": "YAML",
    "json": "JSON",
    "sql": "SQL",
}


def _parse_code_filters(query: str) -> tuple[str, str | None, str | None, str | None]:
    """解析可选内联代码过滤器 `lang:py` / `repo:owner/name` / `path:src/`，返回纯净查询词与过滤条件。"""
    lang: str | None = None
    repo: str | None = None
    path_f: str | None = None

    def _consume(m: re.Match[str]) -> str:
        nonlocal lang, repo, path_f
        k = m.group(1).lower()
        v = m.group(2).strip()
        if k == "lang" and v:
            lang = v
        elif k == "repo" and v:
            repo = v
        elif k == "path" and v:
            path_f = v
        return " "

    cleaned = re.sub(r"\s+", " ", _GREP_FILTER_RE.sub(_consume, query)).strip()
    return cleaned or query.strip(), lang, repo, path_f


def parse_grep_app_text(text: str, limit: int) -> list[SearchItem]:
    """解析 grep.app 官方 MCP 返回的纯文本代码片段块。"""
    results: list[SearchItem] = []
    for chunk in re.split(r"(?m)^(?=Repository: )", text):
        if not chunk.startswith("Repository: "):
            continue
        if len(results) >= limit:
            break
        m_repo = re.search(r"(?m)^Repository:\s*(.+)$", chunk)
        m_path = re.search(r"(?m)^Path:\s*(.+)$", chunk)
        m_url = re.search(r"(?m)^URL:\s*(.+)$", chunk)
        repo = sanitize_text(m_repo.group(1).strip()) if m_repo else ""
        fpath = sanitize_text(m_path.group(1).strip()) if m_path else ""
        url = redact_url_secrets(m_url.group(1).strip()) if m_url else ""
        if not url and not repo:
            continue

        snips = list(
            re.finditer(
                r"(?s)--- Snippet \d+ \(Line (\d+)\) ---[^\r\n]*\r?\n(.*?)(?=\r?\n--- Snippet |\Z)",
                chunk,
            )
        )
        if not snips:
            results.append(
                SearchItem(
                    id=f"grep_app:{repo}:{fpath}",
                    domain="code_evidence",
                    source="grep_app",
                    title=f"{repo} — {fpath}" if fpath else repo,
                    url=url,
                    snippet="",
                    modality="code",
                    extra={"repo": repo, "file": fpath, "line": None},
                )
            )
            continue

        for sm in snips:
            if len(results) >= limit:
                break
            line_no = int(sm.group(1))
            preview = sanitize_text(sm.group(2).rstrip())[:400]
            results.append(
                SearchItem(
                    id=f"grep_app:{repo}:{fpath}:{line_no}",
                    domain="code_evidence",
                    source="grep_app",
                    title=f"{repo} — {fpath}:{line_no}" if fpath else repo,
                    url=url,
                    snippet=preview,
                    modality="code",
                    extra={"repo": repo, "file": fpath, "line": line_no},
                )
            )
    return results


class GrepAppEngine:
    """官方 mcp.grep.app JSON-RPC SSE 跨仓代码片段检索（免 Key L0）。"""

    def __init__(self, endpoint: str = "https://mcp.grep.app/", deadline_sec: float = 12.0):
        self.id = "grep_app"
        self.domain = "code_evidence"
        self.tier = "L0"
        self.endpoint = endpoint
        self.deadline_sec = deadline_sec

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        clean_q, lang, repo, path_f = _parse_code_filters(query)
        tool_args: dict[str, Any] = {"query": clean_q}
        if lang:
            tool_args["language"] = [_GREP_LANG_ALIAS.get(lang.lower(), lang)]
        if repo:
            tool_args["repo"] = repo
        if path_f:
            tool_args["path"] = path_f

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "searchGitHub", "arguments": tool_args},
        }
        headers = {
            "User-Agent": f"search-all-in-one/{__version__}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        req_stats = RequestStats(attempted=1)
        try:
            resp = await client.post(self.endpoint, json=body, headers=headers, timeout=self.deadline_sec)
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
        if resp.status_code == 403:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.FORBIDDEN, reason="HTTP 403 forbidden"),
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

        payload: dict[str, Any] | None = None
        for m in re.finditer(r"(?m)^data:\s*(\{.+\})\s*$", resp.text):
            try:
                cand = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and ("result" in cand or "error" in cand or "id" in cand):
                payload = cand
        if payload is None:
            try:
                payload = resp.json()
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason="grep.app SSE payload unparseable"),
            )

        if payload.get("error"):
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=str(payload["error"])),
            )

        res_obj = payload.get("result") or {}
        content_list = res_obj.get("content") or []
        combined_text = "\n".join(str(c.get("text") or "") for c in content_list if isinstance(c, dict) and c.get("type") == "text")

        # 继承原仓 U-191 契约：HTTP 200 + result.isError=True 必须按限流/错误分档，绝不洗白为 0 命中
        if res_obj.get("isError"):
            req_stats.failed = 1
            err_msg = re.sub(r"\s+", " ", combined_text).strip() or "upstream MCP isError=true"
            is_rl = bool(re.search(r"too many|429|rate.?limit", err_msg, re.IGNORECASE))
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(
                    reason_code=ReasonCode.RATE_LIMITED if is_rl else ReasonCode.ERROR,
                    reason=err_msg[:200],
                ),
            )

        req_stats.succeeded = 1
        items = parse_grep_app_text(combined_text, limit=max(1, limit))
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=items,
            elapsed_ms=elapsed_ms,
            requests=req_stats,
        )


class SourcegraphEngine:
    """Sourcegraph 官方 streaming SSE 跨仓代码片段检索（免 Key L0，可选 SRC_TOKEN 提额）。"""

    def __init__(self, env: dict[str, str] | None = None, deadline_sec: float = 12.0):
        self.id = "sourcegraph"
        self.domain = "code_evidence"
        self.tier = "L0"
        self.deadline_sec = deadline_sec
        self._env = dict(os.environ) if env is None else env

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    def credential_fingerprint(self) -> str:
        from omni_search.health import make_credential_fingerprint

        return make_credential_fingerprint(self._env.get("SRC_TOKEN", ""))

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        clean_q, lang, repo, path_f = _parse_code_filters(query)
        sg_q = f"context:global {clean_q}"
        if lang:
            sg_q += f" lang:{lang}"
        if repo:
            sg_q += f" repo:{re.escape(repo)}$"
        if path_f:
            sg_q += f" path:{re.escape(path_f)}"

        url = "https://sourcegraph.com/.api/search/stream"
        params = {"q": sg_q, "v": "V3", "t": "", "display": str(max(1, limit))}
        headers = {"User-Agent": f"search-all-in-one/{__version__}", "Accept": "text/event-stream"}
        tok = self._env.get("SRC_TOKEN", "").strip()
        if tok:
            headers["Authorization"] = f"token {tok}"

        req_stats = RequestStats(attempted=1)
        results: list[SearchItem] = []
        sse_err: str | None = None
        eff_limit = max(1, int(limit))

        def _consume_frame(ev_name: str, ev_data: str) -> bool:
            nonlocal sse_err
            if ev_name == "error":
                sse_err = ev_data[:160]
                return True
            if ev_name == "done":
                return True
            if ev_name != "matches":
                return False
            try:
                arr = json.loads(ev_data)
            except json.JSONDecodeError:
                return False
            if not isinstance(arr, list):
                return False
            for item in arr:
                if len(results) >= eff_limit:
                    return True
                if not isinstance(item, dict) or item.get("type") != "content":
                    continue
                line_no: int | None = None
                preview = ""
                lmatches = item.get("lineMatches") or []
                cmatches = item.get("chunkMatches") or []
                if lmatches and isinstance(lmatches[0], dict):
                    line_no = int(lmatches[0].get("lineNumber", 0)) + 1
                    preview = sanitize_text(str(lmatches[0].get("line") or "")).strip()[:400]
                elif cmatches and isinstance(cmatches[0], dict):
                    cm = cmatches[0]
                    cstart = cm.get("contentStart") or {}
                    line_no = int(cstart.get("line", 0)) + 1 if isinstance(cstart, dict) else 1
                    preview = sanitize_text(str(cm.get("content") or "")).strip()[:400]
                if line_no is None:
                    continue
                repo_name = sanitize_text(str(item.get("repository") or ""))
                fpath = sanitize_text(str(item.get("path") or ""))
                blob_url = f"https://sourcegraph.com/{repo_name}/-/blob/{quote(fpath, safe='/')}#L{line_no}"
                results.append(
                    SearchItem(
                        id=f"sourcegraph:{repo_name}:{fpath}:{line_no}",
                        domain="code_evidence",
                        source="sourcegraph",
                        title=f"{repo_name} — {fpath}:{line_no}",
                        url=blob_url,
                        snippet=preview,
                        modality="code",
                        extra={"repo": repo_name, "file": fpath, "line": line_no},
                    )
                )
            return len(results) >= eff_limit

        try:
            async with client.stream("GET", url, params=params, headers=headers, timeout=self.deadline_sec) as resp:
                if resp.status_code == 429:
                    req_stats.failed = 1
                    return EngineEnvelope(
                        engine=self.id, domain=self.domain, elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=req_stats, failure=EngineFailure(reason_code=ReasonCode.RATE_LIMITED, reason="HTTP 429 rate-limited"),
                    )
                if resp.status_code == 403:
                    req_stats.failed = 1
                    return EngineEnvelope(
                        engine=self.id, domain=self.domain, elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=req_stats, failure=EngineFailure(reason_code=ReasonCode.FORBIDDEN, reason="HTTP 403 forbidden / edge block"),
                    )
                if resp.status_code >= 400:
                    req_stats.failed = 1
                    return EngineEnvelope(
                        engine=self.id, domain=self.domain, elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=req_stats, failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"HTTP {resp.status_code}"),
                    )
                cur_ev = ""
                data_buf: list[str] = []
                async for raw_line in resp.aiter_lines():
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        if cur_ev and data_buf:
                            if _consume_frame(cur_ev, "\n".join(data_buf)):
                                break
                        cur_ev = ""
                        data_buf = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        cur_ev = line[6:].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line[5:].strip())
                else:
                    if cur_ev and data_buf and len(results) < eff_limit:
                        _consume_frame(cur_ev, "\n".join(data_buf))
        except httpx.TimeoutException as exc:
            if not results:
                req_stats.failed = 1
                return EngineEnvelope(
                    engine=self.id, domain=self.domain, elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=req_stats, failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason=f"timeout: {exc}"),
                )
        except httpx.HTTPError as exc:
            if not results:
                req_stats.failed = 1
                return EngineEnvelope(
                    engine=self.id, domain=self.domain, elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=req_stats, failure=EngineFailure(reason_code=ReasonCode.NETWORK_UNAVAILABLE, reason=str(exc)),
                )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if sse_err is not None and not results:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id, domain=self.domain, elapsed_ms=elapsed_ms, requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"sourcegraph SSE error: {sse_err}"),
            )

        req_stats.succeeded = 1
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=results,
            elapsed_ms=elapsed_ms,
            requests=req_stats,
        )


class GitHubCodeEngine:
    """GitHub 官方 Code Search API（需 GITHUB_TOKEN，缺 token 诚实返回 MISSING_CREDENTIAL）。"""

    def __init__(self, env: dict[str, str] | None = None, deadline_sec: float = 12.0):
        self.id = "github_code"
        self.domain = "code_evidence"
        self.tier = "L2"
        self.deadline_sec = deadline_sec
        self._env = dict(os.environ) if env is None else env

    def credential_status(self) -> tuple[bool, str | None]:
        if not self._env.get("GITHUB_TOKEN", "").strip():
            return False, "missing env GITHUB_TOKEN"
        return True, None

    def credential_fingerprint(self) -> str:
        from omni_search.health import make_credential_fingerprint

        return make_credential_fingerprint(self._env.get("GITHUB_TOKEN", ""))

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        ready, msg = self.credential_status()
        if not ready:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(reason_code=ReasonCode.MISSING_CREDENTIAL, reason=msg or "missing GITHUB_TOKEN"),
            )

        clean_q, lang, repo, path_f = _parse_code_filters(query)
        gh_q = clean_q
        if lang:
            gh_q += f" language:{lang}"
        if repo:
            gh_q += f" repo:{repo}"
        if path_f:
            gh_q += f" path:{path_f}"

        headers = {
            "User-Agent": "omni-search/0.2",
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": f"Bearer {self._env['GITHUB_TOKEN'].strip()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        req_stats = RequestStats(attempted=1)
        try:
            resp = await client.get(
                "https://api.github.com/search/code",
                params={"q": gh_q, "per_page": str(max(1, min(limit, 20)))},
                headers=headers,
                timeout=self.deadline_sec,
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
        if resp.status_code == 401:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.MISSING_CREDENTIAL, reason="HTTP 401 unauthorized GITHUB_TOKEN"),
            )
        if resp.status_code in (403, 429):
            req_stats.failed = 1
            rc = ReasonCode.RATE_LIMITED if (resp.status_code == 429 or "rate limit" in resp.text.lower()) else ReasonCode.FORBIDDEN
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=req_stats,
                failure=EngineFailure(reason_code=rc, reason=f"HTTP {resp.status_code}"),
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

        data = resp.json()
        results: list[SearchItem] = []
        for idx, it in enumerate((data.get("items") or [])[:limit]):
            if not isinstance(it, dict):
                continue
            repo_obj = it.get("repository") or {}
            repo_name = sanitize_text(str(repo_obj.get("full_name") or ""))
            fpath = sanitize_text(str(it.get("path") or ""))
            html_url = redact_url_secrets(str(it.get("html_url") or ""))
            frags = [
                sanitize_text(str(tm.get("fragment") or "")).strip()
                for tm in (it.get("text_matches") or [])
                if isinstance(tm, dict) and tm.get("fragment")
            ]
            snippet = " | ".join(frags)[:400]
            results.append(
                SearchItem(
                    id=f"github_code:{repo_name}:{fpath}:{idx}",
                    domain="code_evidence",
                    source="github_code",
                    title=f"{repo_name} — {fpath}" if repo_name else fpath,
                    url=html_url,
                    snippet=snippet,
                    modality="code",
                    extra={"repo": repo_name, "file": fpath, "line": None},
                )
            )

        req_stats.succeeded = 1
        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=results,
            elapsed_ms=elapsed_ms,
            requests=req_stats,
        )
