"""local_index: 基于 SQLite FTS5 (BM25) 的本地多模态增量搜索引擎（零额外编译依赖）。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import pathlib
import re
import sqlite3
import time
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from pydantic import BaseModel, Field

from omni_search.contracts import (
    EngineEnvelope,
    EngineFailure,
    ReasonCode,
    RequestStats,
    SearchItem,
    sanitize_text,
)
from omni_search.embeddings import (
    Embedder,
    build_default_embedder,
    dot_similarity,
    pack_vector,
    unpack_vector,
)
from omni_search.local_parsers import (
    extract_doc_links,
    extract_epub_with_chunks,
    fast_content_hash,
    parse_markdown_frontmatter,
    split_csv_with_header_chunks,
)
from omni_search.local_query import (  # noqa: F401 - re-exported for backward compatibility
    _ASCII_TOK_RE,
    _CJK_RE,
    _CODE_EXTS,
    _DOC_EXTS,
    _MAX_TEXT_BYTES,
    _build_fts_queries,
    _connect_db,
    _extract_query_snippet,
    _migrate_db,
    _query_db_sync,
    _resolve_related_docs,
    _score_chunk_rows,
    tokenize_bilingual,
)

_IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", ".mypy_cache", ".ruff_cache", ".workbuddy", ".idea", ".vscode",
    ".omni-monitors",
}

_OFFICE_EXTS = {".docx", ".pptx", ".xlsx"}
_EPUB_EXTS = {".epub"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".heic"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
_PDF_EXTS = {".pdf"}


@dataclass
class _ExtractedDoc:
    text: str
    metadata_only: bool
    chunks: list[tuple[str, str]]
    tags: list[str] = field(default_factory=list)
    scanned_pdf: bool = False
    links: list[tuple[str, str]] = field(default_factory=list)


class IndexStats(BaseModel):
    db_path: str
    roots: list[str] = Field(default_factory=list)
    collection: str = "default"
    indexed: int = 0
    skipped_unchanged: int = 0
    removed: int = 0
    metadata_only: int = 0
    errors: int = 0
    total_docs: int = 0
    embedder: str = ""
    vectors_updated: int = 0
    chunks_indexed: int = 0
    elapsed_ms: int = 0


def classify_modality(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in _CODE_EXTS:
        return "code"
    if ext in _DOC_EXTS or ext in _OFFICE_EXTS or ext in _PDF_EXTS or ext in _EPUB_EXTS:
        return "document"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    return "file"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS docs (
            path TEXT PRIMARY KEY,
            root TEXT NOT NULL,
            title TEXT NOT NULL,
            modality TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            metadata_only INTEGER NOT NULL,
            excerpt TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            collection TEXT NOT NULL DEFAULT 'default',
            tags TEXT NOT NULL DEFAULT '',
            scanned_pdf INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            path UNINDEXED,
            title_tok,
            body_tok,
            tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS doc_vectors (
            path TEXT PRIMARY KEY,
            embedder_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doc_vectors_embedder ON doc_vectors(embedder_id);
        CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            locator TEXT NOT NULL,
            text_preview TEXT NOT NULL,
            embedder_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_path ON doc_chunks(path);
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedder ON doc_chunks(embedder_id);
        CREATE TABLE IF NOT EXISTS doc_links (
            src_path TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            link_kind TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doc_links_src ON doc_links(src_path);
        """
    )
    _migrate_db(conn)


from omni_search.local_chunking import (  # noqa: E402,F401
    _FILTER_TOKEN_RE,
    _KNOWN_MODALITIES,
    _compute_line_contexts,
    _path_matches_filters,
    _slide_sort_key,
    _split_char_chunks,
    _split_line_chunks,
    parse_local_filters,
    parse_local_query,
)


def _extract_ooxml_with_chunks(path: pathlib.Path, ext: str) -> tuple[str, list[tuple[str, str]]]:
    """零依赖抽取 .docx/.pptx/.xlsx 正文并按页/段切块（带防 Zip Bomb 字节上限）。"""
    all_parts: list[str] = []
    chunks: list[tuple[str, str]] = []
    total_read = 0
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if ext == ".docx":
            targets = [n for n in names if n in ("word/document.xml", "word/header1.xml", "word/footer1.xml")]
        elif ext == ".pptx":
            targets = sorted(
                (n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")),
                key=_slide_sort_key,
            )
        else:
            targets = [n for n in names if n == "xl/sharedStrings.xml"]
        for slide_idx, name in enumerate(targets[:100], start=1):
            info = zf.getinfo(name)
            if info.file_size > _MAX_TEXT_BYTES or total_read + info.file_size > _MAX_TEXT_BYTES * 2:
                continue
            raw_xml = zf.read(name)
            total_read += len(raw_xml)
            root = ET.fromstring(raw_xml)
            part_texts = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
            joined = sanitize_text(" ".join(part_texts)).strip()
            if not joined:
                continue
            all_parts.append(joined)
            if ext == ".pptx":
                chunks.extend(_split_char_chunks(joined, f"S{slide_idx}", size=1400, step=1200, max_chunks=4))
    full_text = sanitize_text(" ".join(all_parts))[:_MAX_TEXT_BYTES]
    if ext != ".pptx" and full_text:
        chunks = _split_char_chunks(full_text, "C", size=1200, step=1000, max_chunks=32)
    return full_text, chunks


def _read_file_with_chunks(path: pathlib.Path, modality: str) -> _ExtractedDoc:
    """提取文件正文、结构化分块、Frontmatter 标签、扫描件 PDF 标识与一跳引用边。"""
    ext = path.suffix.lower()
    if ext in _CODE_EXTS or ext in _DOC_EXTS:
        raw = path.read_bytes()[:_MAX_TEXT_BYTES]
        text = sanitize_text(raw.decode("utf-8", errors="replace"))
        tags: list[str] = []
        if ext in (".md", ".markdown"):
            _, tags = parse_markdown_frontmatter(text)
        if ext in (".csv", ".tsv"):
            chunks = split_csv_with_header_chunks(text) or [("L1-L1", f"{path.name} {text[:800]}")]
        else:
            chunks = _split_line_chunks(text) or [("L1-L1", f"{path.name} {text[:800]}")]
        links = extract_doc_links(path, text)
        return _ExtractedDoc(text=text, metadata_only=False, chunks=chunks, tags=tags, links=links)
    if ext in _OFFICE_EXTS:
        try:
            txt, chunks = _extract_ooxml_with_chunks(path, ext)
            if txt.strip():
                return _ExtractedDoc(text=txt, metadata_only=False, chunks=chunks or [("C1", txt[:1200])])
        except Exception:
            pass
    if ext in _EPUB_EXTS:
        try:
            txt, chunks = extract_epub_with_chunks(path, max_bytes=_MAX_TEXT_BYTES)
            if txt.strip():
                return _ExtractedDoc(text=txt, metadata_only=False, chunks=chunks or [("CH1", txt[:1400])])
        except Exception:
            pass
    scanned_pdf = False
    scanned_pages = 0
    if ext in _PDF_EXTS:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            num_pages = len(reader.pages)
            page_texts: list[str] = []
            chunks_pdf: list[tuple[str, str]] = []
            for p_idx, pg in enumerate(reader.pages[:40], start=1):
                ptxt = sanitize_text(pg.extract_text() or "").strip()
                if not ptxt:
                    continue
                page_texts.append(ptxt)
                chunks_pdf.extend(_split_char_chunks(ptxt, f"P{p_idx}", size=1400, step=1200, max_chunks=4))
            txt = "\n".join(page_texts).strip()
            if len(txt) >= 30:
                return _ExtractedDoc(
                    text=txt[:_MAX_TEXT_BYTES],
                    metadata_only=False,
                    chunks=chunks_pdf or [("P1", txt[:1400])],
                )
            if num_pages > 0:
                scanned_pdf = True
                scanned_pages = num_pages
        except Exception:
            pass
    # 图像/音视频/二进制或无文本层扫描件 PDF：提取纯 Python 媒体头元数据并诚实降级
    stat = path.stat()
    media_text = ""
    if modality in ("image", "audio", "video"):
        from omni_search.media_meta import extract_media_metadata, format_media_metadata_text

        media_text = format_media_metadata_text(extract_media_metadata(path))
    parts = [path.name, path.parent.name, modality, ext]
    if scanned_pdf:
        parts.append(f"[scanned-pdf-no-ocr: {scanned_pages} pages]")
    if media_text:
        parts.append(media_text)
    parts.append(f"{stat.st_size} bytes")
    meta = " ".join(p for p in parts if p)
    return _ExtractedDoc(text=meta, metadata_only=True, chunks=[("META", meta)], scanned_pdf=scanned_pdf)


def index_paths(
    roots: list[pathlib.Path | str],
    db_path: pathlib.Path | str,
    reset: bool = False,
    embedder: Embedder | None = None,
    collection: str = "default",
) -> IndexStats:
    """扫描指定目录并增量更新本地 SQLite FTS5 + 向量混合索引（支持 SHA-256 哈希去重与多 collection 命名空间）。"""
    t0 = time.perf_counter()
    db = pathlib.Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    if reset and db.exists():
        db.unlink()

    eff_collection = (collection or "default").strip().lower() or "default"
    emb = embedder or build_default_embedder()
    norm_roots = [pathlib.Path(r).resolve() for r in roots]
    stats = IndexStats(
        db_path=str(db),
        roots=[str(r) for r in norm_roots],
        collection=eff_collection,
        embedder=emb.id,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = _connect_db(db)
    try:
        _init_db(conn)
        cur = conn.cursor()

        for root in norm_roots:
            if not root.exists():
                stats.errors += 1
                continue
            root_str = str(root)
            existing: dict[str, tuple[int, int, str, str]] = {
                row[0]: (int(row[1]), int(row[2]), str(row[3] or ""), str(row[4] or "default"))
                for row in cur.execute(
                    "SELECT path, mtime_ns, size_bytes, content_hash, collection FROM docs WHERE root = ?",
                    (root_str,),
                )
            }
            existing_vecs: dict[str, str] = {
                row[0]: str(row[1])
                for row in cur.execute(
                    "SELECT v.path, v.embedder_id FROM doc_vectors v JOIN docs d ON d.path = v.path WHERE d.root = ?",
                    (root_str,),
                )
            }
            existing_chunks: dict[str, str] = {
                row[0]: str(row[1])
                for row in cur.execute(
                    "SELECT DISTINCT c.path, c.embedder_id FROM doc_chunks c JOIN docs d ON d.path = c.path WHERE d.root = ?",
                    (root_str,),
                )
            }
            seen: set[str] = set()
            pending_doc_chunks: list[tuple[str, str, list[tuple[str, str]]]] = []

            file_iter = [root] if root.is_file() else _walk_files(root, skip_db=db.resolve())
            for fpath in file_iter:
                pstr = str(fpath)
                seen.add(pstr)
                try:
                    st = fpath.stat()
                except OSError:
                    stats.errors += 1
                    continue

                prev = existing.get(pstr)
                same_col = bool(prev and prev[3] == eff_collection)
                mtime_unchanged = bool(prev and prev[0] == st.st_mtime_ns and prev[1] == st.st_size and same_col)
                vecs_ready = existing_vecs.get(pstr) == emb.id and existing_chunks.get(pstr) == emb.id
                if mtime_unchanged and vecs_ready:
                    stats.skipped_unchanged += 1
                    continue

                # SHA-256 快速内容哈希校验：若 mtime 抖动但文件大小与哈希未变，跳过重解析与向量重算
                cur_hash = fast_content_hash(fpath, st.st_size)
                hash_unchanged = bool(
                    prev and same_col and prev[1] == st.st_size and prev[2] and cur_hash and prev[2] == cur_hash
                )
                if hash_unchanged and vecs_ready:
                    cur.execute("UPDATE docs SET mtime_ns = ? WHERE path = ?", (st.st_mtime_ns, pstr))
                    stats.skipped_unchanged += 1
                    continue

                unchanged = mtime_unchanged or hash_unchanged
                modality = classify_modality(fpath)
                try:
                    extracted = _read_file_with_chunks(fpath, modality)
                except Exception:
                    stats.errors += 1
                    continue

                pending_doc_chunks.append((pstr, f"{fpath.name} {fpath.parent.name}", extracted.chunks))

                if unchanged:
                    # 文件内容未变，仅因切换嵌入器或老库升级补算分块向量
                    if hash_unchanged and not mtime_unchanged:
                        cur.execute("UPDATE docs SET mtime_ns = ? WHERE path = ?", (st.st_mtime_ns, pstr))
                    stats.skipped_unchanged += 1
                    continue

                if extracted.metadata_only:
                    stats.metadata_only += 1
                excerpt = re.sub(r"\s+", " ", extracted.text).strip()[:400]
                tags_csv = ",".join(extracted.tags)
                title_tok = tokenize_bilingual(fpath.name)
                tag_ctx = " ".join(extracted.tags)
                body_tok = tokenize_bilingual(f"{fpath.parent.name} {tag_ctx} {extracted.text[:60000]}")

                cur.execute(
                    """
                    INSERT OR REPLACE INTO docs
                        (path, root, title, modality, size_bytes, mtime_ns, metadata_only, excerpt, indexed_at,
                         content_hash, collection, tags, scanned_pdf)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pstr,
                        root_str,
                        fpath.name,
                        modality,
                        st.st_size,
                        st.st_mtime_ns,
                        1 if extracted.metadata_only else 0,
                        excerpt,
                        now_iso,
                        cur_hash,
                        eff_collection,
                        tags_csv,
                        1 if extracted.scanned_pdf else 0,
                    ),
                )
                cur.execute("DELETE FROM docs_fts WHERE path = ?", (pstr,))
                cur.execute(
                    "INSERT INTO docs_fts (path, title_tok, body_tok) VALUES (?, ?, ?)",
                    (pstr, title_tok, body_tok),
                )
                cur.execute("DELETE FROM doc_links WHERE src_path = ?", (pstr,))
                if extracted.links:
                    cur.executemany(
                        "INSERT INTO doc_links (src_path, target_ref, link_kind) VALUES (?, ?, ?)",
                        [(pstr, ref, kind) for ref, kind in extracted.links],
                    )
                stats.indexed += 1

            # 展平所有待嵌入 chunk 分批（32 条）计算向量，写入 doc_chunks 与 doc_vectors
            flat_jobs: list[tuple[str, int, str, str, str]] = []
            for pstr, header_ctx, chunks in pending_doc_chunks:
                cur.execute("DELETE FROM doc_chunks WHERE path = ?", (pstr,))
                for c_idx, (loc, c_text) in enumerate(chunks):
                    preview = re.sub(r"\s+", " ", c_text).strip()[:360]
                    embed_txt = f"{header_ctx}\n{c_text}" if c_idx == 0 else c_text
                    flat_jobs.append((pstr, c_idx, loc, preview, embed_txt))

            updated_doc_paths: set[str] = set()
            for i in range(0, len(flat_jobs), 32):
                batch = flat_jobs[i : i + 32]
                try:
                    vecs = emb.embed_texts([item[4] for item in batch])
                except Exception:
                    stats.errors += len(batch)
                    continue
                for (pstr, c_idx, loc, preview, _), vec in zip(batch, vecs):
                    if not vec:
                        continue
                    blob = pack_vector(vec)
                    cid = f"{pstr}#{c_idx}"
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO doc_chunks
                            (chunk_id, path, chunk_index, locator, text_preview, embedder_id, dim, vec)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (cid, pstr, c_idx, loc, preview, emb.id, len(vec), blob),
                    )
                    stats.chunks_indexed += 1
                    if c_idx == 0:
                        cur.execute(
                            "INSERT OR REPLACE INTO doc_vectors (path, embedder_id, dim, vec) VALUES (?, ?, ?, ?)",
                            (pstr, emb.id, len(vec), blob),
                        )
                    updated_doc_paths.add(pstr)
            stats.vectors_updated += len(updated_doc_paths)

            # 清理该根目录下已删除的旧条目
            for stale_path in set(existing.keys()) - seen:
                cur.execute("DELETE FROM docs WHERE path = ?", (stale_path,))
                cur.execute("DELETE FROM docs_fts WHERE path = ?", (stale_path,))
                cur.execute("DELETE FROM doc_vectors WHERE path = ?", (stale_path,))
                cur.execute("DELETE FROM doc_chunks WHERE path = ?", (stale_path,))
                cur.execute("DELETE FROM doc_links WHERE src_path = ?", (stale_path,))
                stats.removed += 1

        conn.commit()
        row = cur.execute("SELECT COUNT(*) FROM docs").fetchone()
        stats.total_docs = int(row[0]) if row else 0
    finally:
        conn.close()

    stats.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return stats


def _walk_files(root: pathlib.Path, skip_db: pathlib.Path | None = None):
    skip_names = {
        ".omni-cache.db", ".omni-cache.db-wal", ".omni-cache.db-shm", ".omni-cache.db-journal",
        ".omni-telemetry.db", ".omni-telemetry.db-wal", ".omni-telemetry.db-shm", ".omni-telemetry.db-journal",
    }
    if skip_db:
        skip_names.update({skip_db.name, f"{skip_db.name}-wal", f"{skip_db.name}-shm", f"{skip_db.name}-journal"})
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS and not d.startswith(".git")]
        base = pathlib.Path(dirpath)
        for fn in filenames:
            fp = base / fn
            if fn in skip_names and (
                skip_db is None
                or fp.parent.resolve() == skip_db.parent
                or fn.startswith((".omni-cache.db", ".omni-telemetry.db"))
            ):
                continue
            yield fp


class LocalIndexEngine:
    """将本地 SQLite FTS5 + 向量混合索引封装为标准 EngineContract v2 引擎节点。"""

    def __init__(
        self,
        db_path: pathlib.Path | str,
        deadline_sec: float = 5.0,
        embedder: Embedder | None = None,
    ):
        self.id = "local_index"
        self.domain = "local"
        self.tier = "L0"
        self.deadline_sec = deadline_sec
        self.db_path = pathlib.Path(db_path)
        self.embedder = embedder or build_default_embedder()

    def doc_count(self) -> int:
        if not self.db_path.exists():
            return 0
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                row = conn.execute("SELECT COUNT(*) FROM docs").fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return 0

    def credential_status(self) -> tuple[bool, str | None]:
        cnt = self.doc_count()
        if cnt <= 0:
            return False, f"local index empty ({self.db_path}); run 'omni index <dir>'"
        return True, f"{cnt} docs indexed ({self.embedder.id})"

    async def search(self, query: str, limit: int, client: Any = None) -> EngineEnvelope:
        t0 = time.perf_counter()
        if not self.db_path.exists():
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(
                    reason_code=ReasonCode.NOT_RUN,
                    reason=f"local index not initialized ({self.db_path}); run 'omni index <dir>'",
                ),
            )
        try:
            total, items, warnings = await asyncio.to_thread(_query_db_sync, self.db_path, query, limit, self.embedder)
        except Exception as exc:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=1, succeeded=0, failed=1),
                failure=EngineFailure(reason_code=ReasonCode.ERROR, reason=str(exc)),
            )

        if total == 0:
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                requests=RequestStats(attempted=0, succeeded=0, failed=0),
                failure=EngineFailure(
                    reason_code=ReasonCode.NOT_RUN,
                    reason=f"local index has 0 docs ({self.db_path}); run 'omni index <dir>'",
                ),
            )

        return EngineEnvelope(
            engine=self.id,
            domain=self.domain,
            results=items,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            requests=RequestStats(attempted=1, succeeded=1, failed=0),
            warnings=warnings,
        )
