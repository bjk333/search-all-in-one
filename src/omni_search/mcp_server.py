"""标准 JSON-RPC 2.0 stdio MCP Server（提供 outputSchema + structuredContent）。"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any, Callable

from omni_search import __version__
from omni_search.cache import DiskCacheStore
from omni_search.code_evidence import GitHubCodeEngine, GrepAppEngine, SourcegraphEngine
from omni_search.contracts import compact_search_response
from omni_search.declarative import DeclarativeEngine, load_declarative_engines
from omni_search.embeddings import build_default_embedder
from omni_search.fetcher import safe_fetch_url, safe_fetch_urls_batch
from omni_search.health import HealthStore
from omni_search.local_code import LocalCodeEngine
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.mcp_schemas import (  # noqa: F401 - re-exported for backward compatibility
    _COMMON_OUTPUT_SCHEMA,
    _PROMPTS_SPEC,
    _RESOURCES_SPEC,
    _TOOLS_SPEC,
)
from omni_search.mcps_bridge import McpsBridgeEngine
from omni_search.monitor import render_monitor_markdown, run_monitor_check
from omni_search.package_engines import GoPkgSearchEngine, PyPiSearchEngine
from omni_search.pool import FederatedPool, SearchEngine
from omni_search.reranker import build_default_reranker
from omni_search.research import compact_research_report, render_research_markdown, run_research_rounds
from omni_search.security_engines import OsvVulnsEngine
from omni_search.site_map import safe_crawl_site, safe_map_site
from omni_search.telemetry import TelemetryStore
from omni_search.verifier import ClaimInput, verify_claims

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_ENGINES_DIR = PROJECT_ROOT / "engines"
DEFAULT_INDEX_DB = PROJECT_ROOT / ".omni-index.db"
DEFAULT_CACHE_DB = PROJECT_ROOT / ".omni-cache.db"
DEFAULT_TELEMETRY_DB = PROJECT_ROOT / ".omni-telemetry.db"


_SHARED_CRED_WHITELIST = {
    "DEEPSEEK_API_KEY",
    "BRAVE_API_KEY",
    "TAVILY_API_KEY",
    "SEARXNG_URL",
    "OPENALEX_API_KEY",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "CODEBERG_TOKEN",
    "GITEE_TOKEN",
    "SRC_TOKEN",
    "NVD_API_KEY",
}


def _read_dotenv_file(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            out[k] = v
    return out


def load_dotenv_into_env(env_path: pathlib.Path | None = None) -> dict[str, str]:
    """只读加载 .env（进程环境优先 -> 本仓 .env -> 可选回退原 mcps .env 白名单凭据）。"""
    merged = dict(os.environ)
    for k, v in _read_dotenv_file(env_path or (PROJECT_ROOT / ".env")).items():
        merged.setdefault(k, v)

    # 不复制密钥落盘，仅在未禁用时按白名单从关联 mcps 仓 .env 只读复用搜索凭据
    if merged.get("OMNI_NO_MCPS_ENV", "").strip() not in ("1", "true", "yes"):
        mcps_script = merged.get("MCPS_SCRIPT_PATH", "").strip()
        from omni_search.mcps_bridge import DEFAULT_MCPS_PATH

        mcps_env = (pathlib.Path(mcps_script) if mcps_script else DEFAULT_MCPS_PATH).parent / ".env"
        for k, v in _read_dotenv_file(mcps_env).items():
            if k in _SHARED_CRED_WHITELIST:
                merged.setdefault(k, v)

    # 若仍未配置 GITHUB_TOKEN，自动复用 GH_TOKEN 或本机已登录的 gh CLI 凭据（仅内存态，不落盘）
    if not merged.get("GITHUB_TOKEN", "").strip():
        from omni_search.contracts import resolve_github_token

        gh_tok = resolve_github_token(merged)
        if gh_tok:
            merged["GITHUB_TOKEN"] = gh_tok
    return merged


def build_default_engines(
    engines_dir: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    index_db_path: pathlib.Path | None = None,
) -> list[SearchEngine]:
    """加载全部原生引擎（声明式 YAML + 跨仓代码片段证据 + 本机混合索引 + 免建库实时代码检索）。"""
    eff_env = load_dotenv_into_env() if env is None else env
    specs = load_declarative_engines(engines_dir or DEFAULT_ENGINES_DIR)
    engines: list[SearchEngine] = [DeclarativeEngine(s, env=eff_env) for s in specs]
    # 原生跨仓公网代码片段证据源（完全替代原仓 grep.app / Sourcegraph / GitHub Code）
    engines.append(GrepAppEngine())
    engines.append(SourcegraphEngine(env=eff_env))
    engines.append(GitHubCodeEngine(env=eff_env))
    # 原生开源漏洞安全源（Google OSV.dev 统一漏洞图谱，与 NVD CVE 形成双源交叉印证）
    engines.append(OsvVulnsEngine())
    # 原生 PyPI / Go 包生态精准检索引擎（替代忽略 q 参数的 Ecosyste.ms /packages 假活端点）
    engines.append(PyPiSearchEngine())
    engines.append(GoPkgSearchEngine())
    # 本地多模态增量全文 + 分块向量混合索引
    db_env = eff_env.get("OMNI_INDEX_DB", "").strip()
    db_path = index_db_path or (pathlib.Path(db_env) if db_env else DEFAULT_INDEX_DB)
    engines.append(LocalIndexEngine(db_path=db_path, embedder=build_default_embedder(eff_env)))
    # 本地免建库实时代码检索（rg / 可选 ast-grep）
    engines.append(LocalCodeEngine(env=eff_env))
    # 可选兼容开关：仅在显式设置 OMNI_ENABLE_MCPS_BRIDGE=1 时挂载旧 PowerShell 桥接
    if eff_env.get("OMNI_ENABLE_MCPS_BRIDGE", "").strip().lower() in ("1", "true", "yes"):
        engines.append(McpsBridgeEngine("mcps_repo", "code", mode="repo", env=eff_env))
        engines.append(McpsBridgeEngine("mcps_packages", "packages", mode="research", env=eff_env))
    return engines


class OmniMCPServer:
    def __init__(
        self,
        engines_dir: pathlib.Path | None = None,
        health_path: pathlib.Path | None = None,
        env: dict[str, str] | None = None,
        emit_notification: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.env = load_dotenv_into_env() if env is None else env
        self._emit_notification = emit_notification
        self._last_research_markdown: str = ""
        db_env = self.env.get("OMNI_INDEX_DB", "").strip()
        self.index_db_path = pathlib.Path(db_env) if db_env else DEFAULT_INDEX_DB
        self.embedder = build_default_embedder(self.env)
        self.reranker = build_default_reranker(self.env)
        cache_ttl = float(self.env.get("OMNI_DISK_CACHE_TTL", "300") or "300")
        cache_env = self.env.get("OMNI_CACHE_DB", "").strip()
        cache_path = pathlib.Path(cache_env) if cache_env else DEFAULT_CACHE_DB
        self.disk_cache = DiskCacheStore(cache_path, ttl_sec=cache_ttl) if cache_ttl > 0 else None
        tel_off = self.env.get("OMNI_TELEMETRY", "1").strip().lower() in ("0", "false", "off")
        tel_env = self.env.get("OMNI_TELEMETRY_DB", "").strip()
        tel_path = pathlib.Path(tel_env) if tel_env else DEFAULT_TELEMETRY_DB
        self.telemetry = None if tel_off else TelemetryStore(tel_path)
        self.engines = build_default_engines(engines_dir=engines_dir, env=self.env, index_db_path=self.index_db_path)
        self.health_store = HealthStore(health_path or (PROJECT_ROOT / ".omni-health.json"))
        self.pool = FederatedPool(
            engines=self.engines,
            health_store=self.health_store,
            reranker=self.reranker,
            disk_cache=self.disk_cache,
            telemetry=self.telemetry,
        )

    async def aclose(self) -> None:
        await self.pool.aclose()

    async def doctor(
        self,
        probe: bool = False,
        reset_health: bool = False,
        reset_engines: list[str] | None = None,
        stats: bool = False,
        stats_days: int = 7,
    ) -> dict[str, Any]:
        reset_applied: str | list[str] | None = None
        if reset_health:
            self.health_store.reset()
            reset_applied = "all"
        elif reset_engines:
            cleaned = [str(eid).strip() for eid in reset_engines if str(eid).strip()]
            if cleaned:
                for eid in cleaned:
                    self.health_store.reset(eid)
                reset_applied = cleaned
        rows: list[dict[str, Any]] = []
        for eng in self.engines:
            ready, msg = eng.credential_status() if hasattr(eng, "credential_status") else (True, None)
            fp = eng.credential_fingerprint() if hasattr(eng, "credential_fingerprint") else ""
            h = self.health_store.check(eng.id, cred_fingerprint=fp)
            rows.append(
                {
                    "id": eng.id,
                    "domain": eng.domain,
                    "tier": eng.tier,
                    "ready": ready,
                    "detail": msg or "ok",
                    "disabled": eng.id.lower() in self.pool.disabled_engines,
                    "in_cooldown": h.in_cooldown,
                    "half_open": h.half_open,
                    "skip_fast": h.skip_fast,
                    "consecutive_failures": h.consecutive_failures,
                    "last_reason": h.last_reason,
                }
            )
        probe_summary = None
        if probe:
            probe_resp = await self.pool.search("python", domains=["knowledge", "news"], limit=1, budget_ms=10000)
            probe_summary = {
                "verdict": probe_resp.coverage.verdict,
                "executed": probe_resp.coverage.executed,
                "degraded": probe_resp.coverage.degraded,
                "elapsed_ms": probe_resp.ledger.elapsed_ms,
            }
        local_eng = next((e for e in self.engines if isinstance(e, LocalIndexEngine)), None)
        indexed_docs = local_eng.doc_count() if local_eng else 0

        health_summary = {
            "ready_count": sum(1 for r in rows if r["ready"] and not r["disabled"]),
            "missing_cred_count": sum(1 for r in rows if not r["ready"]),
            "disabled_count": sum(1 for r in rows if r["disabled"]),
            "cooldown_count": sum(1 for r in rows if r["in_cooldown"]),
            "half_open_count": sum(1 for r in rows if r["half_open"]),
        }
        recommendations: list[str] = []
        quota_ids = [r["id"] for r in rows if r["last_reason"] == "quota-exhausted" and not r["disabled"]]
        if quota_ids:
            q_csv = ",".join(quota_ids)
            recommendations.append(
                f"配额耗尽源 ({', '.join(quota_ids)})：可充值更新凭据后执行 `omni doctor --reset-engines {q_csv}`，或在 .env 配置 `OMNI_DISABLE={q_csv}`"
            )
        chronic_ids = [
            r["id"]
            for r in rows
            if r["consecutive_failures"] >= 4 and r["last_reason"] != "quota-exhausted" and not r["disabled"]
        ]
        if chronic_ids:
            c_csv = ",".join(chronic_ids)
            recommendations.append(
                f"高频失败/半开探测源 ({', '.join(chronic_ids)})：网络恢复后可执行 `omni doctor --reset-engines {c_csv}` 清除历史计数，或按需加入 OMNI_DISABLE"
            )
        if indexed_docs == 0:
            recommendations.append("本地知识库尚无文档：可执行 `omni index <目录路径>` 构建离线混合索引")
        if self.embedder.id.startswith("hash_ngram"):
            recommendations.append("当前使用零依赖哈希子词向量：如需真语义检索，可安装 `uv sync --extra semantic` 并设 `OMNI_EMBED_BACKEND=model2vec` 或配置本机 `OMNI_EMBED_URL`")

        out: dict[str, Any] = {
            "version": __version__,
            "engine_count": len(rows),
            "domains": sorted({e.domain for e in self.engines}),
            "disabled_engines": sorted(self.pool.disabled_engines),
            "reset_applied": reset_applied,
            "health_summary": health_summary,
            "recommendations": recommendations,
            "local_index": {
                "db_path": str(self.index_db_path),
                "indexed_docs": indexed_docs,
                "embedder": self.embedder.id,
            },
            "reranker": self.reranker.id if self.reranker is not None else None,
            "engines": rows,
            "probe": probe_summary,
        }
        if stats:
            out["telemetry"] = (
                self.telemetry.summarize(window_days=stats_days)
                if self.telemetry is not None
                else {"window_days": stats_days, "total_events": 0, "engines": [], "top_contributors": [], "recommendations": []}
            )
        return out

    async def handle_request(
        self,
        req: dict[str, Any],
        emit_notification: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}
        eff_emit = emit_notification if emit_notification is not None else self._emit_notification

        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "tools": {},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {"name": "search-all-in-one", "version": __version__},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS_SPEC}}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": _RESOURCES_SPEC}}
        if method == "resources/read":
            uri = str(params.get("uri") or "")
            if uri == "omni://research/last-report":
                txt = self._last_research_markdown or "# 暂无调研简报\n\n请先调用 `omni_research` 执行一轮深度调研。\n"
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": txt}]},
                }
            if uri == "omni://doctor/telemetry":
                tel = (
                    self.telemetry.summarize(window_days=7)
                    if self.telemetry is not None
                    else {"window_days": 7, "total_events": 0, "engines": [], "top_contributors": [], "recommendations": []}
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(tel, ensure_ascii=False)}]},
                }
            if uri == "omni://local/index/summary":
                summary = self._local_index_summary()
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(summary, ensure_ascii=False)}]},
                }
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"unknown resource uri: {uri}"}}
        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": _PROMPTS_SPEC}}
        if method == "prompts/get":
            pname = str(params.get("name") or "")
            pargs = params.get("arguments") or {}
            msg_text = self._render_prompt_template(pname, pargs)
            if msg_text is None:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"unknown prompt: {pname}"}}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "description": f"Search All in One 调研模板: {pname}",
                    "messages": [{"role": "user", "content": {"type": "text", "text": msg_text}}],
                },
            }
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                if name == "omni_search":
                    resp = await self.pool.search(
                        query=str(args["query"]),
                        domains=args.get("domains"),
                        limit=int(args.get("limit", 6)),
                        max_results=int(args.get("maxResults", 25)),
                        budget_ms=int(args.get("budgetMs", 25000)),
                        auto_relax=bool(args.get("autoRelax", True)),
                    )
                    payload = compact_search_response(resp) if bool(args.get("concise", False)) else resp.model_dump()
                elif name == "omni_fetch":
                    urls_arg = args.get("urls")
                    if isinstance(urls_arg, list) and urls_arg:
                        bres = await safe_fetch_urls_batch(
                            urls=[str(u) for u in urls_arg[:20]],
                            max_chars=int(args.get("maxChars", 8000)),
                            timeout_sec=float(args.get("timeoutSec", 15)),
                            concurrency=int(args.get("concurrency", 5)),
                            disk_cache=self.disk_cache,
                            use_cache=not bool(args.get("noCache", False)),
                            prefer_llms_txt=bool(args.get("preferLlmsTxt", False)),
                            query=str(args.get("query", "") or ""),
                        )
                        payload = bres.model_dump()
                    elif str(args.get("mode", "fetch")).strip().lower() == "crawl":
                        if not args.get("url"):
                            raise ValueError("omni_fetch(mode='crawl') requires 'url'")
                        cres = await safe_crawl_site(
                            url=str(args["url"]),
                            query=str(args.get("query", "") or ""),
                            max_pages=int(args.get("maxPages", 5)),
                            max_chars_per_page=int(args.get("maxChars", 4000)),
                            concurrency=int(args.get("concurrency", 4)),
                            timeout_sec=float(args.get("timeoutSec", 15)),
                            disk_cache=self.disk_cache,
                            use_cache=not bool(args.get("noCache", False)),
                        )
                        payload = cres.model_dump()
                    elif str(args.get("mode", "fetch")).strip().lower() == "map":
                        if not args.get("url"):
                            raise ValueError("omni_fetch(mode='map') requires 'url'")
                        mres = await safe_map_site(
                            url=str(args["url"]),
                            query=str(args.get("query", "") or ""),
                            max_urls=int(args.get("maxUrls", 50)),
                            timeout_sec=float(args.get("timeoutSec", 12)),
                        )
                        payload = mres.model_dump()
                    else:
                        if not args.get("url"):
                            raise ValueError("omni_fetch requires 'url' or 'urls'")
                        res = await safe_fetch_url(
                            url=str(args["url"]),
                            max_chars=int(args.get("maxChars", 8000)),
                            timeout_sec=float(args.get("timeoutSec", 15)),
                            disk_cache=self.disk_cache,
                            use_cache=not bool(args.get("noCache", False)),
                            prefer_llms_txt=bool(args.get("preferLlmsTxt", False)),
                            query=str(args.get("query", "") or ""),
                        )
                        payload = res.model_dump()
                elif name == "omni_batch":
                    queries = [str(q) for q in args["queries"][:16]]
                    tasks = [
                        self.pool.search(
                            query=q,
                            domains=args.get("domains"),
                            limit=int(args.get("limit", 5)),
                            budget_ms=int(args.get("budgetMs", 30000)),
                        )
                        for q in queries
                    ]
                    outs = await asyncio.gather(*tasks)
                    payload = {
                        "version": __version__,
                        "query_count": len(outs),
                        "items": [o.model_dump() for o in outs],
                    }
                elif name == "omni_research":
                    meta = params.get("_meta") or {}
                    prog_token = meta.get("progressToken")

                    def _progress_cb(step: int, total: int, message: str) -> None:
                        if prog_token is not None and eff_emit is not None:
                            eff_emit(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "notifications/progress",
                                    "params": {
                                        "progressToken": prog_token,
                                        "progress": step,
                                        "total": total,
                                        "message": message,
                                    },
                                }
                            )

                    report = await run_research_rounds(
                        pool=self.pool,
                        brief=str(args["brief"]),
                        domains=args.get("domains"),
                        mode=str(args.get("mode", "standard")),
                        extra_terms=args.get("terms"),
                        limit=int(args.get("limit", 4)),
                        verify_top_k=int(args.get("verifyTopK", 3)),
                        budget_ms=int(args.get("budgetMs", 35000)),
                        render_markdown=bool(args.get("renderMarkdown", False)),
                        render_bibtex=bool(args.get("renderBibtex", False)),
                        include_local=bool(args.get("includeLocal", False)),
                        deepen=bool(args.get("deepen", False)),
                        progress_cb=_progress_cb if prog_token is not None else None,
                    )
                    self._last_research_markdown = report.markdown or render_research_markdown(report)
                    payload = compact_research_report(report) if bool(args.get("concise", False)) else report.model_dump()
                elif name == "omni_index":
                    paths = [str(p) for p in args["paths"]]
                    stats = await asyncio.to_thread(
                        index_paths,
                        paths,
                        self.index_db_path,
                        bool(args.get("reset", False)),
                        self.embedder,
                        str(args.get("collection", "default") or "default"),
                    )
                    payload = stats.model_dump()
                elif name == "omni_doctor":
                    raw_re = args.get("resetEngines")
                    re_list = [str(x).strip() for x in raw_re if str(x).strip()] if isinstance(raw_re, list) else None
                    payload = await self.doctor(
                        probe=bool(args.get("probe", False)),
                        reset_health=bool(args.get("resetHealth", False)),
                        reset_engines=re_list,
                        stats=bool(args.get("stats", False)),
                        stats_days=int(args.get("statsDays", 7)),
                    )
                elif name == "omni_monitor":
                    sdir = pathlib.Path(str(args.get("stateDir") or (PROJECT_ROOT / ".omni-monitors")))
                    mres = await run_monitor_check(
                        pool=self.pool,
                        topic=str(args["topic"]),
                        state_dir=sdir,
                        mode=str(args.get("mode", "standard")),
                        domains=args.get("domains"),
                        limit=int(args.get("limit", 4)),
                        verify_top_k=int(args.get("verifyTopK", 2)),
                        budget_ms=int(args.get("budgetMs", 25000)),
                    )
                    payload = mres.model_dump()
                    if bool(args.get("renderMarkdown", False)):
                        payload["markdown"] = render_monitor_markdown(mres)
                elif name == "omni_verify":
                    raw_claims = args.get("claims") or []
                    claim_objs = [ClaimInput(claim=str(c["claim"]), url=str(c["url"])) for c in raw_claims]
                    md_text = str(args.get("markdown") or "")
                    if not claim_objs and not md_text.strip():
                        raise ValueError("omni_verify requires 'markdown' or non-empty 'claims'")
                    vrep = await verify_claims(
                        claims=claim_objs,
                        markdown=md_text or None,
                        concurrency=int(args.get("concurrency", 5)),
                        timeout_sec=float(args.get("timeoutSec", 15)),
                        max_chars=int(args.get("maxChars", 12000)),
                        disk_cache=self.disk_cache,
                        use_cache=not bool(args.get("noCache", False)),
                        render_markdown=bool(args.get("renderMarkdown", False)),
                    )
                    payload = vrep.model_dump()
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": f"unknown tool: {name}"},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                        "structuredContent": payload,
                        "isError": False,
                    },
                }
            except KeyError as exc:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"missing parameter: {exc}"}}
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, ensure_ascii=False)}],
                        "isError": True,
                    },
                }
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"method not found: {method}"}}

    def _local_index_summary(self) -> dict[str, Any]:
        if not self.index_db_path.exists():
            return {"db_path": str(self.index_db_path), "total_docs": 0, "collections": {}, "modalities": {}}
        try:
            conn = sqlite3.connect(str(self.index_db_path), timeout=3.0)
            try:
                total = int(conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
                mods = {str(r[0]): int(r[1]) for r in conn.execute("SELECT modality, COUNT(*) FROM docs GROUP BY modality").fetchall()}
                cols: dict[str, int] = {}
                col_names = {r[1] for r in conn.execute("PRAGMA table_info(docs)").fetchall()}
                if "collection" in col_names:
                    cols = {str(r[0] or "default"): int(r[1]) for r in conn.execute("SELECT collection, COUNT(*) FROM docs GROUP BY collection").fetchall()}
                return {"db_path": str(self.index_db_path), "total_docs": total, "collections": cols, "modalities": mods}
            finally:
                conn.close()
        except Exception as exc:
            return {"db_path": str(self.index_db_path), "error": str(exc)}

    @staticmethod
    def _render_prompt_template(name: str, args: dict[str, Any]) -> str | None:
        if name == "feasibility_research":
            topic = str(args.get("topic") or "").strip() or "<待补充主题>"
            return (
                f"请使用 Search All in One (`omni_research`) 对「{topic}」做可行性深度调研："
                f"设置 mode='feasibility'、deepen=true、verifyTopK=3，核验一手原文并按「证据 / 竞品对比 / 风险与缺口 / 落地建议」输出简体中文报告。"
            )
        if name == "benchmark_comparison":
            topic = str(args.get("topic") or "").strip() or "<待补充主题>"
            return (
                f"请使用 Search All in One (`omni_research`) 对「{topic}」做开源标杆与学术基准对比："
                f"设置 mode='benchmark'、verifyTopK=3，重点核验核心项目 README 与评测指标，标注 T1/T2/T3 证据等级。"
            )
        if name == "security_cve_audit":
            product = str(args.get("product") or "").strip() or "<待补充组件>"
            return (
                f"请使用 Search All in One (`omni_search`) 对「{product}」执行安全 CVE 核查："
                f"检索 domains=['security', 'code', 'news']，并用 `omni_fetch` 核验官方公告或补丁提交，明确披露覆盖缺口。"
            )
        return None

    async def serve_stdio(self) -> None:
        if self._emit_notification is None:
            def _write_notif(payload: dict[str, Any]) -> None:
                sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                sys.stdout.flush()

            self._emit_notification = _write_notif
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
                sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            resp = await self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
