"""security_engines: 原生异步开源安全漏洞引擎（Google OSV.dev 统一漏洞库，免 Key）。"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from omni_search.contracts import (
    EngineEnvelope,
    EngineFailure,
    ReasonCode,
    RequestStats,
    SearchItem,
    sanitize_text,
)

_VULN_ID_RE = re.compile(
    r"\b((?:CVE-\d{4}-\d{4,7}|GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}|PYSEC-\d{4}-\d+|RUSTSEC-\d{4}-\d+|GO-\d{4}-\d+))\b",
    re.IGNORECASE,
)
_ECOSYSTEM_PREFIX_RE = re.compile(r"\b(pypi|pip|npm|crates|cargo|go|golang|maven|nuget|rubygems|packagist):([^\s]+)", re.IGNORECASE)
_ECO_MAP = {
    "pypi": "PyPI",
    "pip": "PyPI",
    "npm": "npm",
    "crates": "crates.io",
    "cargo": "crates.io",
    "go": "Go",
    "golang": "Go",
    "maven": "Maven",
    "nuget": "NuGet",
    "rubygems": "RubyGems",
    "packagist": "Packagist",
}
_BATCH_ECOSYSTEMS = ("PyPI", "npm", "crates.io", "Go")
_SECURITY_NOISE = {
    "cve", "vulnerability", "vulnerabilities", "vuln", "security", "advisory",
    "exploit", "rce", "xss", "ssrf", "overflow", "bug", "patch", "fix", "audit",
    "漏洞", "安全", "补丁", "公告",
}


def _to_search_item(vuln: dict[str, Any], ecosystem: str = "") -> SearchItem:
    vid = sanitize_text(str(vuln.get("id") or "")).strip()
    aliases = [sanitize_text(str(a)).strip() for a in (vuln.get("aliases") or []) if a]
    summary = sanitize_text(str(vuln.get("summary") or vuln.get("details") or "")).strip()
    summary = re.sub(r"\s+", " ", summary)
    prefix = f"[{', '.join(aliases[:3])}] " if aliases else ""
    eco_tag = f"({ecosystem}) " if ecosystem else ""
    snippet = f"{eco_tag}{prefix}{summary}"[:420].strip() or f"{eco_tag}{vid} security advisory"
    title = f"{vid}: {summary[:90]}" if summary else vid
    pub = vuln.get("published") or vuln.get("modified")
    return SearchItem(
        id=vid,
        domain="security",
        source="osv_vulns",
        title=title,
        url=f"https://osv.dev/vulnerability/{vid}",
        snippet=snippet,
        published_at=str(pub) if pub else None,
        modality="advisory",
        extra={"aliases": aliases, "ecosystem": ecosystem} if (aliases or ecosystem) else {},
    )


def _extract_pkg_candidate(query: str) -> str:
    for tok in re.findall(r"[A-Za-z0-9@][A-Za-z0-9._/@-]*", query):
        low = tok.lower()
        if low in _SECURITY_NOISE or len(low) < 2 or low.isdigit():
            continue
        return tok
    return ""


class OsvVulnsEngine:
    """Google OSV.dev 开源漏洞数据库直连引擎（支持 ID 直查、生态前缀包查与多生态批量查）。"""

    id = "osv_vulns"
    domain = "security"
    tier = "L0"

    def __init__(self, deadline_sec: float = 10.0):
        self.deadline_sec = deadline_sec

    def credential_status(self) -> tuple[bool, str | None]:
        return True, None

    async def search(self, query: str, limit: int, client: httpx.AsyncClient) -> EngineEnvelope:
        t0 = time.perf_counter()
        headers = {"User-Agent": "search-all-in-one/0.3", "Accept": "application/json"}
        req_att = req_ok = req_fail = 0

        try:
            # 模式 1：精确漏洞编号直查（CVE / GHSA / PYSEC / RUSTSEC / GO）
            m_id = _VULN_ID_RE.search(query)
            if m_id:
                vid = m_id.group(1)
                if vid.lower().startswith(("cve-", "pysec-", "rustsec-", "go-")):
                    vid = vid.upper()
                elif vid.lower().startswith("ghsa-"):
                    vid = "GHSA-" + vid[5:].lower()
                req_att += 1
                resp = await client.get(
                    f"https://api.osv.dev/v1/vulns/{vid}",
                    headers=headers,
                    timeout=self.deadline_sec,
                )
                if resp.status_code == 200:
                    req_ok += 1
                    data = resp.json()
                    items = [_to_search_item(data)] if isinstance(data, dict) and data.get("id") else []
                    return EngineEnvelope(
                        engine=self.id,
                        domain=self.domain,
                        results=items[:limit],
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                    )
                if resp.status_code == 404:
                    req_ok += 1
                    return EngineEnvelope(
                        engine=self.id,
                        domain=self.domain,
                        results=[],
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                    )
                req_fail += 1
                rc = ReasonCode.RATE_LIMITED if resp.status_code == 429 else ReasonCode.ERROR
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                    failure=EngineFailure(reason_code=rc, reason=f"OSV HTTP {resp.status_code}"),
                )

            # 模式 2：显式生态前缀包查（pypi:urllib3 / npm:lodash 等）
            m_eco = _ECOSYSTEM_PREFIX_RE.search(query)
            if m_eco:
                eco = _ECO_MAP[m_eco.group(1).lower()]
                pkg = m_eco.group(2).strip()
                req_att += 1
                resp = await client.post(
                    "https://api.osv.dev/v1/query",
                    headers=headers,
                    json={"package": {"name": pkg, "ecosystem": eco}},
                    timeout=self.deadline_sec,
                )
                if resp.status_code != 200:
                    req_fail += 1
                    rc = ReasonCode.RATE_LIMITED if resp.status_code == 429 else ReasonCode.ERROR
                    return EngineEnvelope(
                        engine=self.id,
                        domain=self.domain,
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                        failure=EngineFailure(reason_code=rc, reason=f"OSV query HTTP {resp.status_code}"),
                    )
                req_ok += 1
                vulns = resp.json().get("vulns") or []
                vulns_sorted = sorted(vulns, key=lambda v: str(v.get("published") or v.get("modified") or ""), reverse=True)
                items = [_to_search_item(v, ecosystem=eco) for v in vulns_sorted[:limit] if isinstance(v, dict) and v.get("id")]
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    results=items,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                )

            # 模式 3：无前缀包名多生态一次批量探测（PyPI / npm / crates.io / Go）
            pkg_cand = _extract_pkg_candidate(query)
            if not pkg_cand:
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    results=[],
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=0, succeeded=0, failed=0),
                )
            req_att += 1
            batch_resp = await client.post(
                "https://api.osv.dev/v1/querybatch",
                headers=headers,
                json={"queries": [{"package": {"name": pkg_cand, "ecosystem": eco}} for eco in _BATCH_ECOSYSTEMS]},
                timeout=self.deadline_sec,
            )
            if batch_resp.status_code != 200:
                req_fail += 1
                rc = ReasonCode.RATE_LIMITED if batch_resp.status_code == 429 else ReasonCode.ERROR
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                    failure=EngineFailure(reason_code=rc, reason=f"OSV batch HTTP {batch_resp.status_code}"),
                )
            req_ok += 1
            batch_results = batch_resp.json().get("results") or []
            matched_eco: str | None = None
            fallback_vulns: list[tuple[str, dict[str, Any]]] = []
            for eco, entry in zip(_BATCH_ECOSYSTEMS, batch_results):
                vs = (entry or {}).get("vulns") or []
                if vs:
                    if matched_eco is None:
                        matched_eco = eco
                    for v in vs[:limit]:
                        if isinstance(v, dict) and v.get("id"):
                            fallback_vulns.append((eco, v))

            if matched_eco is None:
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    results=[],
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
                )

            # 对首个命中生态做一次完整详情查询，失败时平滑回退 batch 摘要
            items: list[SearchItem] = []
            try:
                req_att += 1
                det_resp = await client.post(
                    "https://api.osv.dev/v1/query",
                    headers=headers,
                    json={"package": {"name": pkg_cand, "ecosystem": matched_eco}},
                    timeout=min(self.deadline_sec, 6.0),
                )
                if det_resp.status_code == 200:
                    req_ok += 1
                    vulns = det_resp.json().get("vulns") or []
                    vulns_sorted = sorted(vulns, key=lambda v: str(v.get("published") or v.get("modified") or ""), reverse=True)
                    items = [_to_search_item(v, ecosystem=matched_eco) for v in vulns_sorted[:limit] if isinstance(v, dict) and v.get("id")]
            except Exception:
                pass
            if not items:
                items = [_to_search_item(v, ecosystem=eco) for eco, v in fallback_vulns[:limit]]

            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                results=items[:limit],
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=req_att, succeeded=req_ok, failed=req_fail),
            )
        except httpx.TimeoutException:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=max(1, req_att), succeeded=req_ok, failed=req_fail + 1),
                failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason="OSV request timeout"),
            )
        except httpx.HTTPError as exc:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=max(1, req_att), succeeded=req_ok, failed=req_fail + 1),
                failure=EngineFailure(reason_code=ReasonCode.NETWORK_UNAVAILABLE, reason=str(exc)),
            )
