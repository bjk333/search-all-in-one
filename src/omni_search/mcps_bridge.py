"""桥接现有 multi-code-platform-searcher（复用其 23 个代码/学术/包源）。"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time
from typing import Any

from omni_search.contracts import (
    EngineEnvelope,
    EngineFailure,
    ReasonCode,
    RequestStats,
    SearchItem,
    redact_url_secrets,
    sanitize_text,
)

DEFAULT_MCPS_PATH = pathlib.Path(r"D:\01 项目\skill\multi-code-platform-searcher\mcps.py")

_RC_MAP = {
    "missing-credential": ReasonCode.MISSING_CREDENTIAL,
    "tool-missing": ReasonCode.TOOL_MISSING,
    "network-unavailable": ReasonCode.NETWORK_UNAVAILABLE,
    "rate-limited": ReasonCode.RATE_LIMITED,
    "forbidden": ReasonCode.FORBIDDEN,
    "deadline": ReasonCode.DEADLINE,
    "budget-truncated": ReasonCode.BUDGET_TRUNCATED,
    "user-disabled": ReasonCode.USER_DISABLED,
    "not-run": ReasonCode.NOT_RUN,
    "zero-hit": ReasonCode.ZERO_HIT,
}


def parse_mcps_json_payload(
    engine_id: str,
    domain: str,
    payload: dict[str, Any],
    elapsed_ms: int = 0,
) -> EngineEnvelope:
    """将 multi-code-platform-searcher 的 -AsJson 载荷无损映射为 EngineEnvelope v2。"""
    results: list[SearchItem] = []
    for idx, r in enumerate(payload.get("results") or []):
        if not isinstance(r, dict):
            continue
        platform = sanitize_text(str(r.get("Platform") or engine_id))
        title = sanitize_text(str(r.get("Name") or r.get("Title") or ""))
        url = redact_url_secrets(str(r.get("Url") or ""))
        if not title and not url:
            continue
        desc = sanitize_text(str(r.get("Desc") or r.get("Preview") or ""))[:500]
        pop_raw = r.get("Stars")
        pop = pop_raw if isinstance(pop_raw, (int, float)) else None
        unit = sanitize_text(str(r.get("Unit") or "none"))
        extra: dict[str, Any] = {"platform": platform}
        if r.get("License"):
            extra["license"] = sanitize_text(str(r["License"]))
        if r.get("File"):
            extra["file"] = sanitize_text(str(r["File"]))
        if r.get("Line") is not None:
            extra["line"] = r["Line"]
        if r.get("Archived") is not None:
            extra["archived"] = bool(r["Archived"])

        results.append(
            SearchItem(
                id=f"{engine_id}-{platform}-{idx}",
                domain=domain,
                source=f"{engine_id}:{platform}",
                title=title or url,
                url=url,
                snippet=desc,
                published_at=str(r.get("Updated")) if r.get("Updated") else None,
                modality="code" if r.get("File") else "text",
                popularity=pop,
                unit=unit,
                extra=extra,
            )
        )

    cov = payload.get("coverage") or {}
    verdict = str(cov.get("Verdict") or "full")
    rc_map = cov.get("ReasonCodes") or {}
    warnings = [sanitize_text(str(w)) for w in (payload.get("warnings") or [])]

    failure: EngineFailure | None = None
    partial = False
    if verdict in ("degraded", "partial", "none"):
        picked_rc = ReasonCode.ERROR
        reason_parts: list[str] = []
        for eng_name, code_str in rc_map.items():
            mapped = _RC_MAP.get(str(code_str), ReasonCode.ERROR)
            if mapped not in (ReasonCode.ZERO_HIT, ReasonCode.USER_DISABLED):
                picked_rc = mapped
                reason_parts.append(f"{eng_name}={code_str}")
        if reason_parts or verdict == "none":
            partial = len(results) > 0
            failure = EngineFailure(
                reason_code=picked_rc,
                reason="; ".join(reason_parts) if reason_parts else f"mcps verdict={verdict}",
            )

    succeeded = 1 if (results or failure is None) else 0
    return EngineEnvelope(
        engine=engine_id,
        domain=domain,
        results=results,
        partial=partial,
        elapsed_ms=elapsed_ms,
        requests=RequestStats(attempted=1, succeeded=succeeded, failed=1 - succeeded),
        failure=failure,
        warnings=warnings,
    )


class McpsBridgeEngine:
    """异步子进程桥接现有 multi-code-platform-searcher（不改动原仓任何文件）。"""

    def __init__(
        self,
        engine_id: str,
        domain: str,
        mode: str,
        script_path: pathlib.Path | None = None,
        deadline_sec: float = 28.0,
        env: dict[str, str] | None = None,
    ):
        self.id = engine_id
        self.domain = domain
        self.tier = "L0"
        self.mode = mode  # "repo" | "code" | "research"
        self.deadline_sec = deadline_sec
        self._env = dict(os.environ) if env is None else env
        configured = self._env.get("MCPS_SCRIPT_PATH", "").strip()
        self.script_path = script_path or (pathlib.Path(configured) if configured else DEFAULT_MCPS_PATH)
        self._last_proc: asyncio.subprocess.Process | None = None

    def credential_status(self) -> tuple[bool, str | None]:
        if not self.script_path.exists():
            return False, f"mcps.py not found at {self.script_path}"
        return True, None

    async def search(self, query: str, limit: int, client: Any = None) -> EngineEnvelope:
        t0 = time.perf_counter()
        if not self.script_path.exists():
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                failure=EngineFailure(
                    reason_code=ReasonCode.TOOL_MISSING,
                    reason=f"mcps.py not found: {self.script_path}",
                ),
            )

        args = [sys.executable, str(self.script_path)]
        if self.mode == "repo":
            args.extend(["-Keyword", query, "-Platform", "All", "-MinStars", "0", "-Limit", str(limit), "-AsJson"])
        elif self.mode == "code":
            args.extend(["-CodeEvidence", "-Keyword", query, "-Limit", str(limit), "-AsJson"])
        else:
            args.extend(["-ResearchSources", "-Keyword", query, "-Limit", str(limit), "-AsJson"])

        env = dict(self._env)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.script_path.parent),
                env=env,
            )
            self._last_proc = proc
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.deadline_sec)
        except asyncio.TimeoutError:
            await _cleanup_subprocess(proc)
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason=f"mcps bridge timeout > {self.deadline_sec}s"),
            )
        except asyncio.CancelledError:
            await _cleanup_subprocess(proc)
            raise
        except Exception as exc:
            await _cleanup_subprocess(proc)
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"mcps bridge exec error: {exc}"),
            )
        finally:
            _close_transport_safely(proc)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        text = stdout_b.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
        except Exception as exc:
            err_preview = stderr_b.decode("utf-8", errors="replace").strip()[:200]
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=elapsed_ms,
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"mcps json parse error: {exc}; stderr={err_preview}"),
            )
        return parse_mcps_json_payload(self.id, self.domain, payload, elapsed_ms=elapsed_ms)


def _close_transport_safely(proc: asyncio.subprocess.Process | None) -> None:
    """显式关闭子进程 transport，避免 Windows ProactorEventLoop 在 __del__ 时抛 closed pipe 异常。"""
    if proc is None:
        return
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


async def _cleanup_subprocess(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None:
        return
    try:
        if proc.returncode is None:
            proc.kill()
        await asyncio.shield(asyncio.wait_for(proc.wait(), timeout=3.0))
    except BaseException:
        pass
    finally:
        _close_transport_safely(proc)
