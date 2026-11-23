"""原生免建库本机代码实时检索引擎（基于 ripgrep `rg --json` + 可选 `ast-grep` 语法树查询）。"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import shutil
import time
from typing import Any

from omni_search.contracts import (
    EngineEnvelope,
    EngineFailure,
    ReasonCode,
    RequestStats,
    SearchItem,
    sanitize_text,
)
from omni_search.mcps_bridge import _cleanup_subprocess, _close_transport_safely

_LOCAL_CODE_FILTER_RE = re.compile(r"(?:^|\s)(-?path|lang|glob|symbol):([^\s]+)", re.IGNORECASE)
_SAFE_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_SYMBOL_DEF_PREFIX = (
    r"\b(?:async\s+def|def|class|func|fn|struct|enum|trait|interface|type|const|let|var|"
    r"export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const))\s+"
)
_LANG_TO_GLOB = {
    "py": "*.py",
    "python": "*.py",
    "js": "*.js",
    "ts": "*.ts",
    "tsx": "*.tsx",
    "rs": "*.rs",
    "rust": "*.rs",
    "go": "*.go",
    "java": "*.java",
    "ps1": "*.ps1",
    "md": "*.md",
    "yaml": "*.yaml",
    "yml": "*.yml",
    "json": "*.json",
}


def _sanitize_code_path(raw: str) -> str:
    """规范化 path: 过滤器为安全相对路径（拒绝 `..` 穿越与绝对路径）。"""
    norm = raw.strip().replace("\\", "/")
    if not norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        return ""
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        return ""
    return "/".join(parts)


def _parse_local_code_query(query: str) -> tuple[str, bool, list[str], list[str]]:
    """解析 `ast:` 前缀与 `lang:`/`glob:`/`path:`/`-path:`/`symbol:` 内联过滤，返回 (模式串, 是否AST模式, glob列表, 安全相对路径列表)。"""
    globs: list[str] = []
    paths: list[str] = []
    symbols: list[str] = []

    def _consume(m: re.Match[str]) -> str:
        k = m.group(1).lower()
        v = m.group(2).strip()
        if k == "lang" and v:
            globs.append(_LANG_TO_GLOB.get(v.lower(), f"*.{v.lower()}"))
        elif k == "glob" and v:
            globs.append(v)
        elif k == "path" and v:
            safe_p = _sanitize_code_path(v)
            if safe_p:
                paths.append(safe_p)
        elif k == "-path" and v:
            for part in v.split(","):
                safe_p = _sanitize_code_path(part)
                if safe_p:
                    globs.append(f"!{safe_p}/**")
                    globs.append(f"!**/{safe_p}/**")
        elif k == "symbol" and v:
            if _SAFE_SYMBOL_RE.fullmatch(v):
                symbols.append(v)
        return " "

    cleaned = re.sub(r"\s+", " ", _LOCAL_CODE_FILTER_RE.sub(_consume, query)).strip()
    is_ast = False
    if cleaned.lower().startswith("ast:"):
        is_ast = True
        cleaned = cleaned[4:].strip()
    elif symbols and not cleaned:
        sym_alt = "|".join(re.escape(s) for s in symbols)
        cleaned = f"{_SYMBOL_DEF_PREFIX}(?:{sym_alt})\\b"
    return cleaned, is_ast, globs, paths


class LocalCodeEngine:
    """免建库实时本机代码检索（默认走 `rg --json`，`ast:` 前缀走 `ast-grep`/`sg`）。"""

    def __init__(
        self,
        roots: list[pathlib.Path | str] | None = None,
        rg_bin: str | None = None,
        sg_bin: str | None = None,
        deadline_sec: float = 8.0,
        env: dict[str, str] | None = None,
    ):
        self.id = "local_code"
        self.domain = "local_code"
        self.tier = "L0"
        self.deadline_sec = deadline_sec
        self._env = dict(os.environ) if env is None else env
        if roots is not None:
            self.roots = [pathlib.Path(r).resolve() for r in roots]
        else:
            raw_root = self._env.get("OMNI_LOCAL_CODE_ROOT", "").strip()
            self.roots = [pathlib.Path(raw_root).resolve()] if raw_root else [pathlib.Path.cwd().resolve()]
        self._rg_bin = rg_bin or shutil.which("rg")
        self._sg_bin = sg_bin if sg_bin is not None else (shutil.which("sg") or shutil.which("ast-grep"))

    def credential_status(self) -> tuple[bool, str | None]:
        if not self._rg_bin or not shutil.which(self._rg_bin):
            return False, "ripgrep (rg) not found in PATH"
        ast_note = "rg + ast-grep" if (self._sg_bin and shutil.which(self._sg_bin)) else "rg ready (ast-grep optional)"
        return True, ast_note

    def _resolve_scan_targets(self, rel_paths: list[str]) -> tuple[list[pathlib.Path], list[str]]:
        """将安全相对 path: 过滤解析为受根边界保护的子目录或附加 --glob 条件。"""
        base_roots = [r for r in self.roots if r.exists()]
        if not rel_paths:
            return base_roots, []
        targeted: list[pathlib.Path] = []
        extra_globs: list[str] = []
        for rp in rel_paths:
            matched_dir = False
            for root in base_roots:
                cand = (root / rp).resolve()
                try:
                    cand.relative_to(root.resolve())
                except ValueError:
                    continue
                if cand.exists():
                    targeted.append(cand)
                    matched_dir = True
            if not matched_dir:
                extra_globs.append(f"**/{rp}/**")
        return (targeted or base_roots), extra_globs

    async def search(self, query: str, limit: int, client: Any = None) -> EngineEnvelope:
        t0 = time.perf_counter()
        pattern, is_ast, globs, rel_paths = _parse_local_code_query(query)
        if not pattern:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason="empty code pattern"),
            )

        scan_roots, path_globs = self._resolve_scan_targets(rel_paths)
        all_globs = globs + path_globs

        if is_ast:
            sg_exe = shutil.which(self._sg_bin) if self._sg_bin else None
            if not sg_exe:
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=0, succeeded=0, failed=0),
                    failure=EngineFailure(
                        reason_code=ReasonCode.TOOL_MISSING,
                        reason="ast-grep (sg) not found in PATH for ast: query",
                    ),
                )
            cmd = [sg_exe, "--pattern", pattern, "--json", *(str(r) for r in scan_roots)]
        else:
            rg_exe = shutil.which(self._rg_bin) if self._rg_bin else None
            if not rg_exe:
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    requests=RequestStats(attempted=0, succeeded=0, failed=0),
                    failure=EngineFailure(reason_code=ReasonCode.TOOL_MISSING, reason="ripgrep (rg) not found in PATH"),
                )
            cmd = [
                rg_exe,
                "--json",
                "--smart-case",
                "--max-count",
                "3",
                "--glob",
                "!.git/**",
                "--glob",
                "!.venv/**",
                "--glob",
                "!node_modules/**",
            ]
            for g in all_globs:
                cmd.extend(["--glob", g])
            cmd.append("--")
            cmd.append(pattern)
            cmd.extend(str(r) for r in scan_roots)

        proc = None
        req_stats = RequestStats(attempted=1)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.deadline_sec)
        except asyncio.TimeoutError:
            await _cleanup_subprocess(proc)
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.DEADLINE, reason=f"local code search timeout > {self.deadline_sec}s"),
            )
        except asyncio.CancelledError:
            await _cleanup_subprocess(proc)
            raise
        except Exception as exc:
            await _cleanup_subprocess(proc)
            req_stats.failed = 1
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=req_stats,
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=str(exc)),
            )
        finally:
            _close_transport_safely(proc)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        text = stdout_b.decode("utf-8", errors="replace")
        results: list[SearchItem] = []

        if is_ast:
            try:
                arr = json.loads(text) if text.strip() else []
            except json.JSONDecodeError:
                arr = []
            for idx, m in enumerate(arr[:limit]):
                if not isinstance(m, dict):
                    continue
                fpath = str(m.get("file") or "")
                line_no = int(((m.get("range") or {}).get("start") or {}).get("line", 0)) + 1
                snippet = sanitize_text(str(m.get("text") or "")).strip()[:400]
                p = pathlib.Path(fpath)
                try:
                    file_url = p.resolve().as_uri()
                except Exception:
                    file_url = "file:///" + fpath.replace("\\", "/")
                results.append(
                    SearchItem(
                        id=f"local_code:{fpath}:{line_no}:{idx}",
                        domain="local_code",
                        source="local_code:ast-grep",
                        title=f"{p.name}:{line_no}",
                        url=file_url,
                        snippet=snippet,
                        modality="code",
                        extra={"path": fpath, "file": fpath, "line": line_no, "tool": "ast-grep"},
                    )
                )
        else:
            # rg 退出码：0=有命中，1=0 命中（正常），2=真正报错
            if proc.returncode not in (0, 1):
                err_txt = stderr_b.decode("utf-8", errors="replace").strip()[:200]
                req_stats.failed = 1
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=elapsed_ms,
                    requests=req_stats,
                    failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=f"rg exit {proc.returncode}: {err_txt}"),
                )
            for line in text.splitlines():
                if len(results) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "match":
                    continue
                data = ev.get("data") or {}
                fpath = str((data.get("path") or {}).get("text") or "")
                line_no = int(data.get("line_number") or 0)
                line_txt = sanitize_text(str((data.get("lines") or {}).get("text") or "")).strip()[:400]
                p = pathlib.Path(fpath)
                try:
                    file_url = p.resolve().as_uri()
                except Exception:
                    file_url = "file:///" + fpath.replace("\\", "/")
                results.append(
                    SearchItem(
                        id=f"local_code:{fpath}:{line_no}",
                        domain="local_code",
                        source="local_code:rg",
                        title=p.name,
                        url=file_url,
                        snippet=f"L{line_no}: {line_txt}",
                        modality="code",
                        extra={"path": fpath, "file": fpath, "line": line_no, "tool": "rg"},
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
