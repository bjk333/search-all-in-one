"""fetch_extract: 响应编码嗅探、Agentic 响应头解析、HTML/PDF/Markdown 正文抽取与误返守卫。"""

from __future__ import annotations

import codecs
import io
import re
from typing import Any
from urllib.parse import urlsplit

from omni_search.contracts import sanitize_text
from omni_search.fetch_security import _MAX_PDF_PAGES
from omni_search.html_markdown import (
    _CleanTextExtractor,
    _normalize_extracted_text,
    _unescape_html_entities,
)

_CHARSET_HDR_RE = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9._-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(
    rb"<meta[^>]+charset\s*=\s*['\"]?\s*([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)

_SLUG_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "your", "into", "about",
    "blog", "post", "posts", "article", "articles", "docs", "doc", "page", "pages",
    "html", "htm", "php", "aspx", "index", "news", "guide", "guides",
})
_MD_TOP_NAV_RE = re.compile(r"\]\(https?://[^/\s)]+/[a-z0-9_-]{2,24}/?(?:\?[^)]*)?\)", re.IGNORECASE)


def _detect_text_encoding(raw_bytes: bytes, content_type: str = "", fallback_encoding: str | None = None) -> str:
    """优先使用 Content-Type 显式 charset；缺失时嗅探 BOM 与 HTML <meta charset>（兼容 GBK/GB18030 中文站点）。"""
    m_hdr = _CHARSET_HDR_RE.search(content_type or "")
    if m_hdr:
        cand = m_hdr.group(1).strip()
        try:
            codecs.lookup(cand)
            return cand
        except LookupError:
            pass

    if raw_bytes.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"

    head = raw_bytes[:2048]
    m_meta = _META_CHARSET_RE.search(head)
    if m_meta:
        cand = m_meta.group(1).decode("ascii", errors="ignore").strip().lower()
        if cand in ("gb2312", "gbk"):
            cand = "gb18030"
        try:
            codecs.lookup(cand)
            return cand
        except LookupError:
            pass

    if fallback_encoding:
        try:
            codecs.lookup(fallback_encoding)
            return fallback_encoding
        except LookupError:
            pass
    return "utf-8"


def _parse_agentic_headers(headers: Any) -> tuple[dict[str, str], int | None]:
    """解析 2026 Agent-Ready Web 标准响应头：Content-Signal（AI 消费策略）与 x-markdown-tokens（token 预估）。"""
    signals: dict[str, str] = {}
    raw_sig = str(headers.get("content-signal") or "").strip() if headers else ""
    if raw_sig:
        for part in raw_sig.split(","):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            k_clean = sanitize_text(k).strip().lower()
            v_clean = sanitize_text(v).strip().lower()
            if k_clean and v_clean and len(signals) < 12:
                signals[k_clean] = v_clean

    md_tokens: int | None = None
    raw_tok = str(headers.get("x-markdown-tokens") or "").strip() if headers else ""
    if raw_tok.isdigit():
        val = int(raw_tok)
        if val >= 0:
            md_tokens = val
    return signals, md_tokens


def _extract_path_slug_tokens(url_str: str) -> list[str]:
    """从深度路径 URL（>=2 段）提取末段 slug 的实质词元；根路径或浅路径返回空列表。"""
    parts = urlsplit(url_str)
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 2:
        return []
    candidate = segs[-1]
    if candidate.isdigit() and len(segs) >= 3:
        candidate = segs[-2]
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[-_./%]+", candidate.lower()):
        tok = raw.strip()
        if len(tok) < 3 or tok.isdigit() or tok in _SLUG_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    return tokens if len(tokens) >= 2 else []


def _count_slug_hits(slug_tokens: list[str], text: str) -> int:
    low = (text or "").lower()
    return sum(1 for tok in slug_tokens if tok in low)


def _looks_like_site_root_llms_fallback(url_str: str, title: str, body: str) -> bool:
    """检测深度路径在 Accept: text/markdown 下是否被站点网关误返了根 /llms.txt 首页替身。"""
    slug_tokens = _extract_path_slug_tokens(url_str)
    if not slug_tokens or not body:
        return False
    low = body.lower()
    body_hits = _count_slug_hits(slug_tokens, low)
    title_hits = _count_slug_hits(slug_tokens, title)
    hit_ratio = body_hits / len(slug_tokens)
    title_ratio = title_hits / len(slug_tokens)
    has_llms_marker = ("llms.txt" in low) or ("llms-full.txt" in low) or ("src=llms-txt" in low)
    if has_llms_marker and title_hits == 0 and hit_ratio < 0.75:
        return True
    if has_llms_marker and hit_ratio < 0.5 and title_ratio < 0.5:
        return True
    return body_hits == 0 and len(_MD_TOP_NAV_RE.findall(body)) >= 3


def _extract_html_title_and_body(text_raw: str) -> tuple[str, str]:
    """从原始 HTML 文本中抽取净化后的标题与主体正文。"""
    parser = _CleanTextExtractor()
    parser.feed(text_raw)
    raw_title = "".join(parser.title_parts).strip() or parser.meta_title
    title = re.sub(r"\s+", " ", sanitize_text(_unescape_html_entities(raw_title))).strip()
    article_body = _normalize_extracted_text("".join(parser.article_parts))
    main_body = _normalize_extracted_text("".join(parser.main_parts))
    if len(article_body) >= 120:
        body = article_body
    elif main_body:
        body = main_body
    elif article_body:
        body = article_body
    else:
        body = _normalize_extracted_text("".join(parser.text_parts))
    if not body and parser.meta_description:
        body = _normalize_extracted_text(parser.meta_description)
    return title, body


def _extract_pdf_bytes(raw_bytes: bytes, max_pages: int = _MAX_PDF_PAGES) -> tuple[str, str]:
    """从受控大小的 PDF 字节流抽取标题与正文（纯 Python pypdf，防损坏异常外泄）。"""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    title = ""
    meta = getattr(reader, "metadata", None)
    if meta is not None:
        raw_title = getattr(meta, "title", None) or (meta.get("/Title") if hasattr(meta, "get") else None)
        if raw_title:
            title = re.sub(r"\s+", " ", sanitize_text(str(raw_title))).strip()
    parts: list[str] = []
    for pg in reader.pages[:max_pages]:
        txt = pg.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    body = _normalize_extracted_text("\n\n".join(parts))
    return title, body


def _extract_markdown_title(body: str) -> str:
    """从原生 Markdown 文本抽取标题：优先 YAML frontmatter title，其次首个一级标题 `# ...`。"""
    lines = body.splitlines()
    if not lines:
        return ""
    idx = 0
    if lines[0].strip() == "---":
        idx = 1
        while idx < min(len(lines), 40):
            line = lines[idx].strip()
            if line == "---":
                idx += 1
                break
            m_fm = re.match(r"^title\s*:\s*(.+)$", line, re.IGNORECASE)
            if m_fm:
                val = m_fm.group(1).strip().strip("\"'").strip()
                if val:
                    return sanitize_text(val)[:200]
            idx += 1
    for line in lines[idx : idx + 30]:
        s = line.strip()
        if s.startswith("# ") and len(s) > 2:
            return sanitize_text(s[2:].strip())[:200]
    return ""
