"""local_chunking: 本地知识库的结构感知分块（Contextual Chunking）与内联查询过滤语法解析。"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
import pathlib
import re

_CHUNK_BOUNDARY_RE = re.compile(
    r"^(?:#{1,4}\s+\S|(?:async\s+)?def\s+\w+|class\s+\w+|function\s+\w+|export\s+(?:async\s+)?function\s+\w+)"
)
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_TOP_CLASS_RE = re.compile(r"^class\s+(\w+)")
_TOP_DEF_RE = re.compile(r"^(?:async\s+def|def|function|export\s+(?:async\s+)?function)\s+\w+")
_FILTER_TOKEN_RE = re.compile(r"(?:^|\s)(-?path|type|ext|collection|col|tag):([^\s]+)", re.IGNORECASE)
_DATE_FILTER_RE = re.compile(r"(?:^|\s)(after|before):([^\s]+)", re.IGNORECASE)
_KNOWN_MODALITIES = {"code", "document", "image", "audio", "video", "file"}


def _split_char_chunks(
    text: str,
    locator_prefix: str,
    size: int = 1200,
    step: int = 1000,
    max_chunks: int = 32,
) -> list[tuple[str, str]]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [(locator_prefix, clean)]
    out: list[tuple[str, str]] = []
    pos = 0
    idx = 1
    while pos < len(clean) and len(out) < max_chunks:
        seg = clean[pos : pos + size].strip()
        if seg:
            loc = locator_prefix if locator_prefix.startswith(("P", "S")) else f"{locator_prefix}{idx}"
            out.append((loc, seg))
            idx += 1
        if pos + size >= len(clean):
            break
        pos += step
    return out


def _compute_line_contexts(lines: list[str]) -> list[str]:
    """O(N) 预扫描每行所处的上级 Markdown 标题层级或代码顶层 class 作用域（零 LLM Contextual Chunking）。"""
    contexts: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    current_class = ""
    in_fence = False

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence

        ctx_parts: list[str] = [h for _, h in heading_stack]
        if current_class:
            ctx_parts.append(current_class)
        contexts.append(" > ".join(ctx_parts))

        if in_fence:
            continue

        hm = _MD_HEADING_RE.match(stripped)
        if hm:
            lvl = len(hm.group(1))
            title = hm.group(2).rstrip("#").strip()[:80]
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            if title:
                heading_stack.append((lvl, title))
            continue

        if raw and not raw[0].isspace():
            cm = _TOP_CLASS_RE.match(raw)
            if cm:
                current_class = f"class {cm.group(1)}"
            elif _TOP_DEF_RE.match(raw):
                current_class = ""

    return contexts


def _split_line_chunks(
    text: str,
    lines_per_chunk: int = 40,
    step_lines: int = 32,
    max_chunks: int = 64,
) -> list[tuple[str, str]]:
    """结构感知 + 上下文面包屑分块：在标题/函数边界切块，并为深处分块自动注入上级标题/类上下文。"""
    lines = text.splitlines()
    if not lines:
        return []
    line_ctx = _compute_line_contexts(lines)
    out: list[tuple[str, str]] = []
    start = 0
    n = len(lines)
    while start < n and len(out) < max_chunks:
        end = min(n, start + lines_per_chunk)
        split_at = end
        for i in range(start + 6, end):
            if _CHUNK_BOUNDARY_RE.match(lines[i].strip()):
                split_at = i
                break
        seg = "\n".join(lines[start:split_at]).strip()
        if seg:
            ctx = line_ctx[start]
            if ctx and ctx not in seg[:160]:
                seg = f"[ctx: {ctx}]\n{seg}"
            out.append((f"L{start + 1}-L{split_at}", seg))
        if split_at >= n:
            break
        start = split_at if split_at < end else (start + step_lines)
    return out


def _slide_sort_key(name: str) -> tuple[int, str]:
    m = re.search(r"slide(\d+)\.xml$", name)
    return (int(m.group(1)) if m else 999999, name)


def _sanitize_rel_path_filter(raw: str) -> str:
    """规范化 path: 过滤器为正斜杠相对片段，拒绝 `..` 路径穿越与绝对路径。"""
    norm = raw.strip().replace("\\", "/")
    if not norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        return ""
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        return ""
    return "/".join(parts)


def parse_local_query_full(
    query: str,
) -> tuple[str, set[str], set[str], set[str], set[str], set[str], set[str]]:
    """解析内联过滤语法 `type:` / `ext:` / `collection:`(`col:`) / `tag:` / `path:` / `-path:`。"""
    modalities: set[str] = set()
    exts: set[str] = set()
    collections: set[str] = set()
    tags: set[str] = set()
    paths: set[str] = set()
    excluded_paths: set[str] = set()

    def _consume(m: re.Match[str]) -> str:
        kind = m.group(1).lower()
        raw_val = m.group(2).strip()
        for part in raw_val.split(","):
            v_raw = part.strip()
            v = v_raw.lower()
            if not v:
                continue
            if kind == "type":
                if v in _KNOWN_MODALITIES:
                    modalities.add(v)
                else:
                    exts.add(v if v.startswith(".") else f".{v}")
            elif kind == "ext":
                exts.add(v if v.startswith(".") else f".{v}")
            elif kind in ("collection", "col"):
                collections.add(v)
            elif kind == "tag":
                tags.add(v)
            elif kind == "path":
                safe_p = _sanitize_rel_path_filter(v_raw)
                if safe_p:
                    paths.add(safe_p.lower())
            elif kind == "-path":
                safe_p = _sanitize_rel_path_filter(v_raw)
                if safe_p:
                    excluded_paths.add(safe_p.lower())
        return " "

    cleaned = _FILTER_TOKEN_RE.sub(_consume, query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, modalities, exts, collections, tags, paths, excluded_paths


def parse_local_query_extended(query: str) -> tuple[str, set[str], set[str], set[str], set[str], set[str]]:
    """解析内联过滤语法（含 `path:`），返回 6 元组（向后兼容）。"""
    cleaned, modalities, exts, collections, tags, paths, _ = parse_local_query_full(query)
    return cleaned, modalities, exts, collections, tags, paths


def parse_local_query(query: str) -> tuple[str, set[str], set[str], set[str], set[str]]:
    """解析内联过滤语法 `type:` / `ext:` / `collection:`(`col:`) / `tag:`，返回 5 元组（向后兼容）。"""
    cleaned, modalities, exts, collections, tags, _ = parse_local_query_extended(query)
    return cleaned, modalities, exts, collections, tags


def parse_local_filters(query: str) -> tuple[str, set[str], set[str]]:
    """向后兼容的 3 元组过滤解析入口（type/ext）。"""
    cleaned, modalities, exts, _, _ = parse_local_query(query)
    return cleaned, modalities, exts


def _parse_date_bound(raw: str, *, is_end: bool) -> int | None:
    """将 YYYY / YYYY-MM / YYYY-MM-DD 解析为 UTC 纳秒时间戳边界。"""
    s = raw.strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", s)
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else (12 if is_end else 1)
    if not (1 <= month <= 12):
        return None
    max_day = calendar.monthrange(year, month)[1]
    day = int(m.group(3)) if m.group(3) else (max_day if is_end else 1)
    if not (1 <= day <= max_day):
        return None
    if is_end:
        dt = datetime(year, month, day, 23, 59, 59, 999999, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9) + 999
    dt = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def parse_local_date_bounds(query: str) -> tuple[str, int | None, int | None]:
    """解析并剥离 `after:` / `before:` 时间过滤条件，返回 (剩余查询, after_ns, before_ns)。"""
    after_ns: int | None = None
    before_ns: int | None = None

    def _consume(m: re.Match[str]) -> str:
        nonlocal after_ns, before_ns
        kind = m.group(1).lower()
        val = m.group(2).strip()
        if kind == "after":
            ts = _parse_date_bound(val, is_end=False)
            if ts is not None:
                after_ns = ts if after_ns is None else max(after_ns, ts)
        elif kind == "before":
            ts = _parse_date_bound(val, is_end=True)
            if ts is not None:
                before_ns = ts if before_ns is None else min(before_ns, ts)
        return " "

    cleaned = re.sub(r"\s+", " ", _DATE_FILTER_RE.sub(_consume, query)).strip()
    return cleaned, after_ns, before_ns


def _path_matches_filters(
    path_str: str,
    modality: str,
    allowed_modalities: set[str],
    allowed_exts: set[str],
    doc_collection: str = "default",
    doc_tags_csv: str = "",
    allowed_collections: set[str] | None = None,
    required_tags: set[str] | None = None,
    required_paths: set[str] | None = None,
    excluded_paths: set[str] | None = None,
    mtime_ns: int | None = None,
    after_ns: int | None = None,
    before_ns: int | None = None,
) -> bool:
    if allowed_modalities and modality not in allowed_modalities:
        return False
    if allowed_exts:
        ext = pathlib.Path(path_str).suffix.lower()
        if ext not in allowed_exts:
            return False
    if allowed_collections and (doc_collection or "default").lower() not in allowed_collections:
        return False
    if required_tags:
        doc_tag_set = {t.strip().lower() for t in (doc_tags_csv or "").split(",") if t.strip()}
        if not required_tags.issubset(doc_tag_set):
            return False
    if required_paths or excluded_paths:
        norm_doc = "/" + path_str.replace("\\", "/").strip("/").lower() + "/"
        if required_paths and not any(f"/{rp}/" in norm_doc for rp in required_paths):
            return False
        if excluded_paths and any(f"/{ep}/" in norm_doc for ep in excluded_paths):
            return False
    if after_ns is not None and (mtime_ns is None or mtime_ns < after_ns):
        return False
    if before_ns is not None and (mtime_ns is None or mtime_ns > before_ns):
        return False
    return True
