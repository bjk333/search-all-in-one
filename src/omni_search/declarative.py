"""声明式 YAML 引擎运行时（支持 JSON / Atom XML / OpenAlex 摘要还原）。"""

from __future__ import annotations

import html
import os
import pathlib
import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import yaml
from pydantic import BaseModel, Field

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

_SEG_RE = re.compile(r"^([^\[\]]+)(?:\[(\*|\d+)\])?$")


def _extract_single(obj: Any, path: str) -> Any:
    path = path.strip()
    if len(path) >= 2 and path[0] == "'" and path[-1] == "'":
        return path[1:-1]
    cur: Any = obj
    for seg in path.split("."):
        if cur is None:
            return None
        m = _SEG_RE.match(seg)
        if not m:
            return None
        key, idx = m.group(1), m.group(2)
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
        if idx is not None:
            if not isinstance(cur, list):
                return None
            if idx == "*":
                rest_idx = path.find(seg) + len(seg)
                rem = path[rest_idx:].lstrip(".")
                if not rem:
                    return cur
                return [_extract_single(elem, rem) for elem in cur]
            i = int(idx)
            cur = cur[i] if 0 <= i < len(cur) else None
    return cur


def _apply_transform(val: Any, name: str) -> Any:
    name = name.strip()
    if name == "reconstruct_abstract":
        if not isinstance(val, dict):
            return ""
        positions: list[tuple[int, str]] = []
        for word, idxs in val.items():
            if isinstance(idxs, list):
                for p in idxs:
                    if isinstance(p, int):
                        positions.append((p, str(word)))
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)[:500]
    if name == "strip_html":
        if not isinstance(val, str):
            return ""
        text = re.sub(r"<[^>]+>", "", val)
        text = html.unescape(text).replace("\xa0", " ")
        return re.sub(r"\s+", " ", text).strip()
    if name == "join_comma":
        if isinstance(val, list):
            return ", ".join(str(x) for x in val if x)
        return str(val or "")
    if name == "collapse_ws":
        return re.sub(r"\s+", " ", str(val or "").replace("\xa0", " ")).strip()
    return val


def extract_path(obj: Any, expr: str) -> Any:
    """支持 `a.b[0].c`、`a.b[*].name`、回退 `x || y` 与管道 `field | transform`。"""
    if not expr:
        return None
    for branch in expr.split("||"):
        parts = branch.split("|")
        base = parts[0].strip()
        val = _extract_single(obj, base)
        for tf in parts[1:]:
            val = _apply_transform(val, tf)
        if val not in (None, "", []):
            return val
    return None


class AuthSpec(BaseModel):
    env: str | None = None
    required: bool = False
    placement: str = "header:Authorization"
    prefix: str = ""


class RateLimitSpec(BaseModel):
    rpm: int = 60
    deadline_sec: float = 12.0


class RequestSpec(BaseModel):
    method: str = "GET"
    url: str
    url_env: str | None = None
    url_suffix: str = ""
    max_query_len: int | None = None
    params: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    json_body: dict[str, Any] | None = None


class ResponseSpec(BaseModel):
    items_path: str = ""
    output_map: dict[str, str] = Field(default_factory=dict)
    url_template: str | None = None
    modality: str = "text"
    score_field: str | None = None
    min_score: float | None = None


class DeclarativeSpec(BaseModel):
    id: str
    domain: str
    tier: str = "L0"
    protocol: str = "http_json"  # http_json | http_atom
    description: str = ""
    auth: AuthSpec = Field(default_factory=AuthSpec)
    rate_limit: RateLimitSpec = Field(default_factory=RateLimitSpec)
    request: RequestSpec
    response: ResponseSpec


def _xml_element_to_dict(elem: ET.Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in elem:
        tag = child.tag.split("}", 1)[-1]
        text = (child.text or "").strip()
        val: Any = _xml_element_to_dict(child) if len(child) > 0 else text
        if tag in out:
            if not isinstance(out[tag], list):
                out[tag] = [out[tag]]
            out[tag].append(val)
        else:
            out[tag] = val
    for k, v in elem.attrib.items():
        out[f"@{k}"] = v
    return out


def _render_template(val: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(val, str):
        if val == "{limit_int}":
            return int(ctx["limit"])
        out = val
        for k, v in ctx.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(val, dict):
        return {k: _render_template(v, ctx) for k, v in val.items()}
    if isinstance(val, list):
        return [_render_template(v, ctx) for v in val]
    return val


class DeclarativeEngine:
    """将 YAML DeclarativeSpec 转为可执行异步检索引擎。"""

    def __init__(self, spec: DeclarativeSpec, env: dict[str, str] | None = None):
        self.spec = spec
        self.id = spec.id
        self.domain = spec.domain
        self.tier = spec.tier
        self.deadline_sec = spec.rate_limit.deadline_sec
        self._env = dict(os.environ) if env is None else env

    def credential_status(self) -> tuple[bool, str | None]:
        """返回 (是否就绪, 缺失说明)。"""
        if self.spec.request.url_env:
            if not self._env.get(self.spec.request.url_env, "").strip():
                return False, f"missing env {self.spec.request.url_env}"
        if self.spec.auth.required and self.spec.auth.env:
            if not self._env.get(self.spec.auth.env, "").strip():
                return False, f"missing env {self.spec.auth.env}"
        return True, None

    def credential_fingerprint(self) -> str:
        from omni_search.health import make_credential_fingerprint

        vals: list[str] = []
        if self.spec.auth.env:
            vals.append(self._env.get(self.spec.auth.env, ""))
        if self.spec.request.url_env:
            vals.append(self._env.get(self.spec.request.url_env, ""))
        return make_credential_fingerprint(*vals)

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        ready, missing_msg = self.credential_status()
        if not ready:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(
                    reason_code=ReasonCode.MISSING_CREDENTIAL,
                    reason=missing_msg or "missing credential",
                ),
            )

        warnings: list[str] = []
        eff_query = query
        if self.spec.request.max_query_len and len(eff_query) > self.spec.request.max_query_len:
            eff_query = eff_query[: self.spec.request.max_query_len]
            warnings.append(
                f"[{self.id}] query 超过 {self.spec.request.max_query_len} 字符上限，已截断"
            )

        ctx = {
            "query": eff_query,
            "query_encoded": quote(eff_query, safe=""),
            "limit": max(1, int(limit)),
        }

        base_url = self.spec.request.url
        if self.spec.request.url_env:
            base_url = self._env.get(self.spec.request.url_env, "").rstrip("/") + self.spec.request.url_suffix
        url = _render_template(base_url, ctx)
        params = _render_template(self.spec.request.params, ctx)
        headers = {
            "User-Agent": f"search-all-in-one/{__version__} (+https://github.com/omni-search)",
            "Accept": "application/json, application/atom+xml, application/xml;q=0.9",
            **_render_template(self.spec.request.headers, ctx),
        }
        json_body = _render_template(self.spec.request.json_body, ctx) if self.spec.request.json_body else None

        # 注入凭据
        if self.spec.auth.env:
            token = self._env.get(self.spec.auth.env, "").strip()
            if token:
                kind, _, field = self.spec.auth.placement.partition(":")
                full_val = f"{self.spec.auth.prefix}{token}"
                if kind == "header":
                    headers[field] = full_val
                elif kind == "query":
                    params[field] = full_val
                elif kind == "json" and isinstance(json_body, dict):
                    json_body[field] = full_val

        req_stats = RequestStats(attempted=1, succeeded=0, failed=0)
        try:
            resp = await client.request(
                method=self.spec.request.method.upper(),
                url=url,
                params=params,
                headers=headers,
                json=json_body,
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
                failure=EngineFailure(reason_code=ReasonCode.NETWORK_UNAVAILABLE, reason=f"http error: {exc}"),
            )

        if resp.status_code == 429:
            req_stats.failed = 1
            retry_after = resp.headers.get("Retry-After", "")
            msg = f"HTTP 429 rate-limited (Retry-After={retry_after})" if retry_after else "HTTP 429 rate-limited"
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.RATE_LIMITED, reason=msg),
            )
        if resp.status_code == 401:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.MISSING_CREDENTIAL, reason="HTTP 401 unauthorized"),
                warnings=warnings,
            )
        if resp.status_code == 402:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.QUOTA_EXHAUSTED, reason="HTTP 402 payment required / insufficient balance"),
                warnings=warnings,
            )
        if resp.status_code == 403:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.FORBIDDEN, reason="HTTP 403 forbidden"),
            )
        if resp.status_code >= 400:
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"HTTP {resp.status_code}"),
            )

        req_stats.succeeded = 1
        try:
            items_raw = self._parse_items(resp)
        except Exception as exc:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"parse error: {exc}"),
            )

        results: list[SearchItem] = []
        omap = self.spec.response.output_map
        dropped = 0
        malformed_urls = 0
        for idx, raw in enumerate(items_raw):
            if len(results) >= limit:
                break
            if not isinstance(raw, dict):
                continue
            if self.spec.response.min_score is not None and self.spec.response.score_field:
                raw_score = extract_path(raw, self.spec.response.score_field)
                try:
                    num_score = float(raw_score) if raw_score is not None else 0.0
                except (TypeError, ValueError):
                    num_score = 0.0
                if num_score < self.spec.response.min_score:
                    dropped += 1
                    continue
            title = sanitize_text(str(extract_path(raw, omap.get("title", "title")) or "")).strip()
            item_url = extract_path(raw, omap.get("url", "url"))
            if not item_url and self.spec.response.url_template:
                item_url = self.spec.response.url_template
                for k, expr in omap.items():
                    v = extract_path(raw, expr)
                    if v is not None:
                        item_url = item_url.replace("{" + k + "}", quote(str(v), safe="/:"))
            item_url = redact_url_secrets(str(item_url or ""))
            if item_url.lower().startswith(("http://", "https://")) and not urlsplit(item_url).hostname:
                malformed_urls += 1
                continue
            if not title and not item_url:
                continue
            item_id = sanitize_text(str(extract_path(raw, omap.get("id", "id")) or f"{self.id}-{idx}"))
            snippet = sanitize_text(str(extract_path(raw, omap.get("snippet", "")) or ""))[:500]
            pub = extract_path(raw, omap.get("published_at", ""))
            pop_raw = extract_path(raw, omap.get("popularity", ""))
            pop: float | int | None = None
            if isinstance(pop_raw, (int, float)):
                pop = pop_raw
            elif isinstance(pop_raw, str):
                try:
                    pop = float(pop_raw) if "." in pop_raw else int(pop_raw)
                except ValueError:
                    pop = None
            unit = sanitize_text(str(extract_path(raw, omap.get("popularity_unit", "'none'")) or "none"))

            results.append(
                SearchItem(
                    id=item_id,
                    domain=self.domain,
                    source=self.id,
                    title=title or item_url,
                    url=item_url,
                    snippet=snippet,
                    published_at=str(pub) if pub else None,
                    modality=self.spec.response.modality,
                    popularity=pop,
                    unit=unit,
                )
            )

        failure: EngineFailure | None = None
        if malformed_urls > 0:
            warnings.append(f"[{self.id}] dropped {malformed_urls} malformed URL(s) without hostname")
        if dropped > 0:
            warnings.append(
                f"[{self.id}] 丢弃 {dropped} 条低分噪音结果（{self.spec.response.score_field} < {self.spec.response.min_score}）"
            )
            if not results:
                failure = EngineFailure(
                    reason_code=ReasonCode.FILTERED,
                    reason=f"returned {dropped} items but all dropped below min_score={self.spec.response.min_score}",
                )

        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=results,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            requests=req_stats,
            failure=failure,
            warnings=warnings,
        )

    def _parse_items(self, resp: httpx.Response) -> list[Any]:
        if self.spec.protocol == "http_atom":
            root = ET.fromstring(resp.text)
            target_tag = self.spec.response.items_path or "entry"
            out = []
            for elem in root.iter():
                if elem.tag.split("}", 1)[-1] == target_tag:
                    out.append(_xml_element_to_dict(elem))
            return out
        data = resp.json()
        if not self.spec.response.items_path:
            return data if isinstance(data, list) else []
        extracted = extract_path(data, self.spec.response.items_path)
        return extracted if isinstance(extracted, list) else []


def load_declarative_engines(engines_dir: pathlib.Path) -> list[DeclarativeSpec]:
    """扫描 engines/**/*.yaml 并校验所有声明式引擎规格。"""
    specs: list[DeclarativeSpec] = []
    if not engines_dir.exists():
        return specs
    for p in sorted(engines_dir.rglob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            specs.append(DeclarativeSpec.model_validate(raw))
    return specs
