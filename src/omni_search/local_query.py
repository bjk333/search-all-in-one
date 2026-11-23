"""local_query: 本地 SQLite FTS5 + 分块向量双路检索、RRF 融合、片段定位与一跳图谱解析。"""

from __future__ import annotations

from datetime import datetime, timezone
import pathlib
import re
import sqlite3
from typing import Any

from omni_search.contracts import SearchItem, sanitize_text
from omni_search.embeddings import (
    Embedder,
    build_default_embedder,
    dot_similarity,
    unpack_vector,
)
from omni_search.local_chunking import (
    _path_matches_filters,
    parse_local_date_bounds,
    parse_local_query,
    parse_local_query_extended,
    parse_local_query_full,
)

_CODE_EXTS = {
    ".py", ".ps1", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rs", ".go",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".sql",
}
_DOC_EXTS = {
    ".md", ".markdown", ".txt", ".rst", ".org", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".tsv", ".xml", ".html", ".htm", ".log",
}
_MAX_TEXT_BYTES = 512 * 1024  # 单文件正文读取上限 512 KiB
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_CAMEL_SPLIT_RE = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])")


def tokenize_bilingual(text: str) -> str:
    """中英双语分词预处理：保留完整英文标识符 + 拆分蛇形/驼峰/缩写子词，中文生成 2-gram 提升 FTS5 召回。"""
    if not text:
        return ""
    tokens: list[str] = []
    for m in _ASCII_TOK_RE.finditer(text):
        orig = m.group(0)
        raw = orig.lower()
        tokens.append(raw)
        camel_expanded = _CAMEL_SPLIT_RE.sub(r"\1\3 \2\4", orig)
        if camel_expanded != orig or "_" in raw or "-" in raw or "." in raw:
            sub_seen: set[str] = {raw}
            for sub in re.split(r"[\s_\-.]+", camel_expanded.lower()):
                if len(sub) >= 2 and sub not in sub_seen:
                    sub_seen.add(sub)
                    tokens.append(sub)
    for m in _CJK_RE.finditer(text):
        seg = m.group(0)
        if len(seg) == 1:
            tokens.append(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.append(seg[i : i + 2])
    return " ".join(tokens)


def _connect_db(db: pathlib.Path) -> sqlite3.Connection:
    """统一打开 SQLite 连接并开启 WAL 并发模式与 busy_timeout，避免 omni watch 写时锁库。"""
    conn = sqlite3.connect(str(db), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _migrate_db(conn: sqlite3.Connection) -> None:
    """幂等升级老版本 .omni-index.db schema，确保新增列与图谱表零报错兼容。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(docs)").fetchall()}
    if not cols:
        return
    migrations = [
        ("content_hash", "ALTER TABLE docs ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"),
        ("collection", "ALTER TABLE docs ADD COLUMN collection TEXT NOT NULL DEFAULT 'default'"),
        ("tags", "ALTER TABLE docs ADD COLUMN tags TEXT NOT NULL DEFAULT ''"),
        ("scanned_pdf", "ALTER TABLE docs ADD COLUMN scanned_pdf INTEGER NOT NULL DEFAULT 0"),
    ]
    changed = False
    for col_name, ddl in migrations:
        if col_name not in cols:
            conn.execute(ddl)
            changed = True
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_collection ON docs(collection)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_links (
            src_path TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            link_kind TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_links_src ON doc_links(src_path)")
    if changed:
        conn.commit()


def _resolve_related_docs(cur: sqlite3.Cursor, src_paths: list[str], max_per_doc: int = 3) -> dict[str, list[dict[str, str]]]:
    """根据 doc_links 表为命中文档解析库内一跳关联文档（相对路径 / wikilink / import）。"""
    if not src_paths:
        return {}
    has_links = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='doc_links'"
    ).fetchone()
    if not has_links:
        return {}
    placeholders = ",".join("?" for _ in src_paths)
    raw_edges = cur.execute(
        f"SELECT src_path, target_ref, link_kind FROM doc_links WHERE src_path IN ({placeholders})",
        src_paths,
    ).fetchall()
    if not raw_edges:
        return {}

    all_docs = cur.execute("SELECT path, title FROM docs").fetchall()
    by_norm_path: dict[str, tuple[str, str]] = {}
    by_name: dict[str, tuple[str, str]] = {}
    by_stem: dict[str, tuple[str, str]] = {}
    for dp, dt in all_docs:
        p_str, t_str = str(dp), str(dt)
        p_obj = pathlib.Path(p_str)
        by_norm_path[str(p_obj).replace("\\", "/").lower()] = (t_str, p_str)
        by_name.setdefault(t_str.lower(), (t_str, p_str))
        by_stem.setdefault(p_obj.stem.lower(), (t_str, p_str))

    related: dict[str, list[dict[str, str]]] = {}
    for sp, tref, kind in raw_edges:
        sp_str = str(sp)
        ref = str(tref).strip()
        if not ref:
            continue
        found: tuple[str, str] | None = None
        # 1) 相对路径直连（优先同目录）
        try:
            cand_rel = str((pathlib.Path(sp_str).parent / ref).resolve()).replace("\\", "/").lower()
            found = by_norm_path.get(cand_rel)
        except Exception:
            found = None
        # 2) 文件名或 stem 匹配（wikilink / Python import）
        if found is None:
            ref_name = pathlib.PurePosixPath(ref.replace("\\", "/")).name.lower()
            found = by_name.get(ref_name) or by_stem.get(pathlib.PurePosixPath(ref_name).stem.lower())
        if found is None or found[1] == sp_str:
            continue
        bucket = related.setdefault(sp_str, [])
        if any(item["path"] == found[1] for item in bucket):
            continue
        bucket.append({"title": found[0], "path": found[1], "kind": str(kind)})
        if len(bucket) >= max_per_doc:
            continue
    return {k: v[:max_per_doc] for k, v in related.items()}


def _score_chunk_rows(
    q_vec: list[float],
    chunk_rows: list[tuple[Any, ...]],
    allowed_paths: set[str] | None,
) -> dict[str, tuple[float, str, str]]:
    """计算分块向量余弦相似度并按文档做 Max-Pool（大库自动启用可选 numpy 批量矩阵点积，零依赖无缝回退纯 Python）。"""
    filtered = [
        (str(pstr), str(locator), str(preview), blob)
        for pstr, locator, preview, blob in chunk_rows
        if allowed_paths is None or str(pstr) in allowed_paths
    ]
    if not filtered:
        return {}

    sims: list[float] | None = None
    if len(filtered) >= 256:
        try:
            import numpy as np  # type: ignore

            dim = len(q_vec)
            q_arr = np.asarray(q_vec, dtype=np.float32)
            mat = np.frombuffer(b"".join(row[3] for row in filtered), dtype=np.float32)
            if mat.size == len(filtered) * dim:
                sims = (mat.reshape(len(filtered), dim) @ q_arr).tolist()
        except Exception:
            sims = None

    best_by_path: dict[str, tuple[float, str, str]] = {}
    if sims is not None:
        for (ps, locator, preview, _), sim in zip(filtered, sims):
            s_val = float(sim)
            prev = best_by_path.get(ps)
            if prev is None or s_val > prev[0]:
                best_by_path[ps] = (s_val, locator, preview)
    else:
        for ps, locator, preview, blob in filtered:
            s_val = dot_similarity(q_vec, unpack_vector(blob))
            prev = best_by_path.get(ps)
            if prev is None or s_val > prev[0]:
                best_by_path[ps] = (s_val, locator, preview)
    return best_by_path


def _extract_query_snippet(
    path: pathlib.Path,
    query: str,
    fallback_excerpt: str,
    max_chars: int = 400,
) -> tuple[str, int | None]:
    """对可读文本/代码文件按查询词定位最佳命中行，返回 (带行号上下文片段, matched_line)。"""
    ext = path.suffix.lower()
    if not query.strip() or (ext not in _CODE_EXTS and ext not in _DOC_EXTS) or not path.exists():
        return fallback_excerpt, None
    try:
        raw = path.read_bytes()[:_MAX_TEXT_BYTES]
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return fallback_excerpt, None

    qlow = query.strip().lower()
    terms = [t for t in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", qlow)]
    if not terms:
        return fallback_excerpt, None

    lines = text.splitlines()
    best_idx = -1
    best_score = 0
    for idx, line in enumerate(lines):
        low = line.lower()
        score = 0
        if qlow in low:
            score += 10
        score += sum(2 for t in terms if t in low)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return fallback_excerpt, None

    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 3)
    win = [f"L{i + 1}: {sanitize_text(lines[i]).strip()}" for i in range(start, end) if lines[i].strip()]
    snippet = " | ".join(win)[:max_chars]
    return (snippet or fallback_excerpt), best_idx + 1


def _build_fts_queries(query: str) -> list[str]:
    toks = [t for t in tokenize_bilingual(query).split() if t]
    if not toks:
        return []
    # 唯一化并加双引号避免 FTS5 特殊字符语法错误
    uniq = list(dict.fromkeys(toks))[:12]
    quoted = ['"' + t.replace('"', '""') + '"' for t in uniq]
    queries = [" AND ".join(quoted)]
    if len(quoted) > 1:
        queries.append(" OR ".join(quoted))
    return queries


def _query_db_sync(
    db: pathlib.Path,
    query: str,
    limit: int,
    embedder: Embedder | None = None,
) -> tuple[int, list[SearchItem], list[str]]:
    emb = embedder or build_default_embedder()
    warnings: list[str] = []
    query_no_dates, after_ns, before_ns = parse_local_date_bounds(query)
    (
        clean_query,
        allowed_modalities,
        allowed_exts,
        allowed_collections,
        required_tags,
        required_paths,
        excluded_paths,
    ) = parse_local_query_full(query_no_dates)
    has_filter = bool(
        allowed_modalities
        or allowed_exts
        or allowed_collections
        or required_tags
        or required_paths
        or excluded_paths
        or after_ns is not None
        or before_ns is not None
    )

    conn = _connect_db(db)
    try:
        _migrate_db(conn)
        cur = conn.cursor()
        row = cur.execute("SELECT COUNT(*) FROM docs").fetchone()
        total = int(row[0]) if row else 0
        if total == 0:
            return 0, [], warnings

        # 若带过滤条件，先确定候选允许集合（保证 SQL 与向量两路都不漏掉过滤后的真实 top-k）
        allowed_paths: set[str] | None = None
        if has_filter:
            allowed_paths = {
                str(p)
                for p, mod, col, tags_csv, mtime_ns in cur.execute(
                    "SELECT path, modality, collection, tags, mtime_ns FROM docs"
                )
                if _path_matches_filters(
                    str(p),
                    str(mod),
                    allowed_modalities,
                    allowed_exts,
                    doc_collection=str(col or "default"),
                    doc_tags_csv=str(tags_csv or ""),
                    allowed_collections=allowed_collections,
                    required_tags=required_tags,
                    required_paths=required_paths,
                    excluded_paths=excluded_paths,
                    mtime_ns=int(mtime_ns) if mtime_ns is not None else None,
                    after_ns=after_ns,
                    before_ns=before_ns,
                )
            }
            if not allowed_paths:
                return total, [], warnings

        cand_k = max(limit * 3, 20)
        fetch_k = max(cand_k * 4, 100) if has_filter else cand_k

        # 仅写了过滤器未写查询词：按最近修改时间直接列出
        if not clean_query and has_filter and allowed_paths:
            chosen_rows = cur.execute(
                "SELECT path, title, modality, size_bytes, mtime_ns, metadata_only, excerpt, collection, tags, scanned_pdf FROM docs ORDER BY mtime_ns DESC"
            ).fetchall()
            filtered_rows = [r for r in chosen_rows if str(r[0]) in allowed_paths][: max(1, limit)]
            related_map = _resolve_related_docs(cur, [str(r[0]) for r in filtered_rows])
            items: list[SearchItem] = []
            for pstr, title, modality, size_b, mtime_ns, meta_only, excerpt, col, tags_csv, scanned_pdf in filtered_rows:
                p_str = str(pstr)
                p = pathlib.Path(p_str)
                try:
                    file_url = p.as_uri()
                except ValueError:
                    file_url = "file:///" + p_str.replace("\\", "/")
                pub = datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc).isoformat()
                tag_list = [t for t in str(tags_csv or "").split(",") if t]
                items.append(
                    SearchItem(
                        id=p_str,
                        domain="local",
                        source="local_index",
                        title=sanitize_text(title),
                        url=file_url,
                        snippet=sanitize_text(excerpt),
                        published_at=pub,
                        modality=modality,
                        popularity=size_b,
                        unit="bytes",
                        extra={
                            "path": p_str,
                            "collection": str(col or "default"),
                            "tags": tag_list,
                            "metadata_only": bool(meta_only),
                            "scanned_pdf_no_ocr": bool(scanned_pdf),
                            "related_docs": related_map.get(p_str, []),
                            "retrieval_mode": "filter",
                            "bm25": None,
                            "vector_score": None,
                            "rrf_score": None,
                            "matched_line": None,
                            "embedder": emb.id,
                        },
                    )
                )
            return total, items, warnings

        # 路 1：FTS5 BM25 词法召回
        fts_queries = _build_fts_queries(clean_query)
        lex_rows: list[Any] = []
        for fq in fts_queries:
            try:
                raw_rows = cur.execute(
                    """
                    SELECT docs_fts.path, bm25(docs_fts, 0.0, 5.0, 1.0) AS rank
                    FROM docs_fts
                    WHERE docs_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (fq, fetch_k),
                ).fetchall()
                lex_rows = (
                    [r for r in raw_rows if str(r[0]) in allowed_paths]
                    if allowed_paths is not None
                    else raw_rows
                )[:cand_k]
            except sqlite3.OperationalError:
                lex_rows = []
            if lex_rows:
                break
        lex_rank: dict[str, tuple[int, float]] = {
            str(p): (idx, float(score)) for idx, (p, score) in enumerate(lex_rows, start=1)
        }

        # 路 2：SQLite 分块向量余弦召回（优先 doc_chunks Max-Pool，老库自动回退 doc_vectors）
        vec_rank: dict[str, tuple[int, float]] = {}
        best_chunk_by_path: dict[str, tuple[float, str, str]] = {}  # path -> (sim, locator, preview)
        has_chunk_table = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='doc_chunks'"
        ).fetchone()
        has_vec_table = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='doc_vectors'"
        ).fetchone()
        if (has_chunk_table or has_vec_table) and clean_query:
            try:
                q_vecs = emb.embed_texts([clean_query])
                q_vec = q_vecs[0] if q_vecs else []
            except Exception as exc:
                q_vec = []
                warnings.append(f"vector embed failed ({emb.id}): {exc}; degraded to lexical FTS5")
            if q_vec:
                used_chunks = False
                if has_chunk_table:
                    chunk_rows = cur.execute(
                        "SELECT path, locator, text_preview, vec FROM doc_chunks WHERE embedder_id = ?",
                        (emb.id,),
                    ).fetchall()
                    if chunk_rows:
                        used_chunks = True
                        best_chunk_by_path = _score_chunk_rows(q_vec, chunk_rows, allowed_paths)
                        scored_vecs = [
                            (sim, ps)
                            for ps, (sim, _, _) in best_chunk_by_path.items()
                            if sim >= emb.min_score
                        ]
                        scored_vecs.sort(key=lambda x: x[0], reverse=True)
                        for idx, (sim, pstr) in enumerate(scored_vecs[:cand_k], start=1):
                            vec_rank[pstr] = (idx, sim)

                if not used_chunks and has_vec_table:
                    scored_vecs_fb: list[tuple[float, str]] = []
                    for pstr, blob in cur.execute(
                        "SELECT path, vec FROM doc_vectors WHERE embedder_id = ?",
                        (emb.id,),
                    ):
                        ps = str(pstr)
                        if allowed_paths is not None and ps not in allowed_paths:
                            continue
                        sim = dot_similarity(q_vec, unpack_vector(blob))
                        if sim >= emb.min_score:
                            scored_vecs_fb.append((sim, ps))
                    scored_vecs_fb.sort(key=lambda x: x[0], reverse=True)
                    for idx, (sim, pstr) in enumerate(scored_vecs_fb[:cand_k], start=1):
                        vec_rank[pstr] = (idx, sim)

        # 双路 Reciprocal Rank Fusion (RRF, k=60)
        all_paths = set(lex_rank.keys()) | set(vec_rank.keys())
        if not all_paths:
            return total, [], warnings

        fused: list[tuple[float, str]] = []
        for pstr in all_paths:
            lr = lex_rank.get(pstr)
            vr = vec_rank.get(pstr)
            rrf = (1.0 / (60.0 + lr[0]) if lr else 0.0) + (1.0 / (60.0 + vr[0]) if vr else 0.0)
            tie = (vr[1] * 1e-4) if vr else 0.0
            fused.append((rrf + tie, pstr))
        fused.sort(key=lambda x: x[0], reverse=True)
        chosen = fused[: max(1, limit)]

        placeholders = ",".join("?" for _ in chosen)
        meta_rows = {
            str(r[0]): r
            for r in cur.execute(
                f"SELECT path, title, modality, size_bytes, mtime_ns, metadata_only, excerpt, collection, tags, scanned_pdf FROM docs WHERE path IN ({placeholders})",
                [p for _, p in chosen],
            ).fetchall()
        }
        related_map = _resolve_related_docs(cur, [p for _, p in chosen])

        items = []
        for rrf_score, pstr in chosen:
            meta = meta_rows.get(pstr)
            if not meta:
                continue
            _, title, modality, size_b, mtime_ns, meta_only, excerpt, col, tags_csv, scanned_pdf = meta
            p = pathlib.Path(pstr)
            try:
                file_url = p.as_uri()
            except ValueError:
                file_url = "file:///" + pstr.replace("\\", "/")
            pub = datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc).isoformat()
            lr = lex_rank.get(pstr)
            vr = vec_rank.get(pstr)
            if lr and vr:
                mode = "hybrid"
            elif lr:
                mode = "lexical"
            else:
                mode = "vector"
            centered_snippet, matched_line = _extract_query_snippet(p, clean_query, sanitize_text(excerpt))
            best_chunk = best_chunk_by_path.get(pstr)
            matched_locator: str | None = best_chunk[1] if best_chunk else (f"L{matched_line}" if matched_line else None)
            matched_page: int | None = None
            if matched_locator and re.fullmatch(r"P\d+", matched_locator):
                matched_page = int(matched_locator[1:])
            if matched_line is None and best_chunk is not None and best_chunk[1] != "META":
                centered_snippet = f"[{best_chunk[1]}] {sanitize_text(best_chunk[2])}"[:400]
            tag_list = [t for t in str(tags_csv or "").split(",") if t]
            items.append(
                SearchItem(
                    id=pstr,
                    domain="local",
                    source="local_index",
                    title=sanitize_text(title),
                    url=file_url,
                    snippet=centered_snippet,
                    published_at=pub,
                    modality=modality,
                    popularity=size_b,
                    unit="bytes",
                    extra={
                        "path": pstr,
                        "collection": str(col or "default"),
                        "tags": tag_list,
                        "metadata_only": bool(meta_only),
                        "scanned_pdf_no_ocr": bool(scanned_pdf),
                        "related_docs": related_map.get(pstr, []),
                        "retrieval_mode": mode,
                        "bm25": round(lr[1], 4) if lr else None,
                        "vector_score": round(vr[1], 4) if vr else None,
                        "rrf_score": round(rrf_score, 5),
                        "matched_line": matched_line,
                        "matched_locator": matched_locator,
                        "matched_page": matched_page,
                        "embedder": emb.id,
                    },
                )
            )
        return total, items, warnings
    finally:
        conn.close()
