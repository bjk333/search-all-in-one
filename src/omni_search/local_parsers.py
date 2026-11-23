"""local_parsers: 零依赖本地文档/代码解析与引用抽取辅助模块（SHA-256、Frontmatter、EPUB、CSV、链接图谱）。"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import pathlib
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from omni_search.contracts import sanitize_text

_HASH_SAMPLE_BYTES = 1024 * 1024  # 快速内容哈希采样上限 1 MiB + 文件大小


def fast_content_hash(path: pathlib.Path, size_bytes: int) -> str:
    """计算文件前 1 MiB + 文件大小的 SHA-256，免疫 mtime 抖动且避免超大文件整盘读取。"""
    h = hashlib.sha256()
    h.update(str(size_bytes).encode("ascii"))
    h.update(b":")
    try:
        with path.open("rb") as fh:
            h.update(fh.read(_HASH_SAMPLE_BYTES))
    except OSError:
        return ""
    return h.hexdigest()[:32]


_FM_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def parse_markdown_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """轻量解析 Markdown 顶部 YAML Frontmatter 中的标量字段与 tags 列表（零额外依赖）。"""
    m = _FM_RE.match(text)
    if not m:
        return {}, []
    block = m.group(1)
    meta: dict[str, str] = {}
    tags: list[str] = []
    in_tags_list = False

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if in_tags_list and line.startswith("-"):
            val = line[1:].strip().strip("\"'").lower()
            if val:
                tags.append(val)
            continue
        in_tags_list = False
        if ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        k = key.strip().lower()
        v = raw_val.strip()
        if k in ("tags", "tag", "keywords"):
            if not v:
                in_tags_list = True
                continue
            inner = v.strip("[]")
            for part in re.split(r"[,，]+", inner):
                t = part.strip().strip("\"'").lower()
                if t:
                    tags.append(t)
        elif v:
            meta[k] = v.strip("\"'")

    return meta, list(dict.fromkeys(tags))


class _PlainXhtmlStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "head"):
            self._skip += 1
        elif t in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "head") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            self.parts.append(data)


def _strip_xhtml(raw_bytes: bytes) -> str:
    parser = _PlainXhtmlStripper()
    parser.feed(raw_bytes.decode("utf-8", errors="replace"))
    return sanitize_text(re.sub(r"\s+", " ", "".join(parser.parts))).strip()


def extract_epub_with_chunks(path: pathlib.Path, max_bytes: int = 512 * 1024) -> tuple[str, list[tuple[str, str]]]:
    """零依赖抽取 .epub 章节正文，按 spine 顺序生成 CH1/CH2... 分块（带防 Zip Bomb 限制）。"""
    all_parts: list[str] = []
    chunks: list[tuple[str, str]] = []
    total_read = 0

    with zipfile.ZipFile(path, "r") as zf:
        namelist = zf.namelist()
        name_set = set(namelist)
        chapter_files: list[str] = []

        # 优先按标准 container.xml -> .opf spine 顺序读取章节
        if "META-INF/container.xml" in name_set:
            try:
                croot = ET.fromstring(zf.read("META-INF/container.xml"))
                opf_path = ""
                for elem in croot.iter():
                    if elem.tag.endswith("rootfile") and elem.attrib.get("full-path"):
                        opf_path = elem.attrib["full-path"]
                        break
                if opf_path and opf_path in name_set:
                    opf_dir = pathlib.PurePosixPath(opf_path).parent
                    oroot = ET.fromstring(zf.read(opf_path))
                    manifest: dict[str, str] = {}
                    for elem in oroot.iter():
                        if elem.tag.endswith("item") and elem.attrib.get("id") and elem.attrib.get("href"):
                            href = elem.attrib["href"]
                            full = str(opf_dir / href) if str(opf_dir) != "." else href
                            manifest[elem.attrib["id"]] = full
                    for elem in oroot.iter():
                        if elem.tag.endswith("itemref") and elem.attrib.get("idref"):
                            target = manifest.get(elem.attrib["idref"])
                            if target and target in name_set:
                                chapter_files.append(target)
            except Exception:
                chapter_files = []

        if not chapter_files:
            chapter_files = sorted(
                n for n in namelist if n.lower().endswith((".xhtml", ".html", ".htm"))
            )

        for ch_idx, cname in enumerate(chapter_files[:60], start=1):
            info = zf.getinfo(cname)
            if info.file_size > max_bytes or total_read + info.file_size > max_bytes * 2:
                continue
            raw = zf.read(cname)
            total_read += len(raw)
            txt = _strip_xhtml(raw)
            if not txt:
                continue
            all_parts.append(txt)
            # 每章按 1400 字切块，首块定位符为 CH<n>
            pos = 0
            sub_idx = 0
            while pos < len(txt) and sub_idx < 6:
                seg = txt[pos : pos + 1400].strip()
                if seg:
                    loc = f"CH{ch_idx}" if sub_idx == 0 else f"CH{ch_idx}.{sub_idx + 1}"
                    chunks.append((loc, seg))
                    sub_idx += 1
                if pos + 1400 >= len(txt):
                    break
                pos += 1200

    full_text = sanitize_text("\n\n".join(all_parts))[:max_bytes]
    return full_text, chunks


def split_csv_with_header_chunks(
    text: str,
    rows_per_chunk: int = 25,
    max_chunks: int = 64,
) -> list[tuple[str, str]]:
    """对 CSV/TSV 按行分块，并将第 1 行表头自动前置到每个数据块，防止深处记录丢失列名语义。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0]
    if len(lines) <= rows_per_chunk + 1:
        return [(f"L1-L{len(lines)}", "\n".join(lines))]
    chunks: list[tuple[str, str]] = []
    start = 1
    n = len(lines)
    while start < n and len(chunks) < max_chunks:
        end = min(n, start + rows_per_chunk)
        body = "\n".join([header] + lines[start:end])
        chunks.append((f"L{start + 1}-L{end}", body))
        start = end
    return chunks


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s#]+)")
_PY_FROM_RE = re.compile(r"^\s*from\s+(\.{0,2}[A-Za-z0-9_.]*)\s+import\s+", re.MULTILINE)
_PY_IMP_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)", re.MULTILINE)
_JS_IMP_RE = re.compile(r"""(?:from|import|require\()\s*['"](\.\.?/[^'"]+)['"]""")


def extract_doc_links(path: pathlib.Path, text: str) -> list[tuple[str, str]]:
    """从 Markdown 与常见代码文件中抽取一跳引用边，返回 [(target_ref, link_kind), ...]。"""
    ext = path.suffix.lower()
    links: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(ref: str, kind: str) -> None:
        clean = ref.strip().replace("\\", "/")
        if not clean or len(clean) > 200:
            return
        key = (clean.lower(), kind)
        if key not in seen:
            seen.add(key)
            links.append((clean, kind))

    if ext in (".md", ".markdown"):
        for m in _WIKILINK_RE.finditer(text):
            _add(m.group(1), "wikilink")
        for m in _MD_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.lower().startswith(("http://", "https://", "mailto:", "data:", "#")):
                continue
            _add(target, "md_link")
    elif ext == ".py":
        for m in _PY_FROM_RE.finditer(text):
            mod = m.group(1).lstrip(".")
            if mod:
                _add(mod.split(".")[-1], "py_import")
        for m in _PY_IMP_RE.finditer(text):
            mod = m.group(1)
            _add(mod.split(".")[-1], "py_import")
    elif ext in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"):
        for m in _JS_IMP_RE.finditer(text):
            _add(m.group(1), "js_import")

    return links[:40]
