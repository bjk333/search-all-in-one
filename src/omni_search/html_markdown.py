"""零依赖 HTML → 轻量结构化 Markdown 抽取与查询聚焦段级裁剪模块。"""

from __future__ import annotations

import html as _html
from html.parser import HTMLParser
import re

_SKIP_TAGS = {"head", "script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "iframe"}
_MAIN_TAGS = {"main", "article"}
_BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "tr", "br", "pre"}
_HEADING_PREFIX = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}


class _CleanTextExtractor(HTMLParser):
    """零依赖 HTML → 轻量结构化 Markdown 抽取器（优先 <article>/<main>，剥离 <svg>/<header>/<nav>）。"""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._header_skip_stack: list[bool] = []
        self._svg_depth = 0
        self._main_depth = 0
        self._article_depth = 0
        self._in_title = False
        self._head_title_done = False
        self._in_pre = False
        self._pre_fence_emitted = False
        self._in_code = False
        self._in_cell = False
        self._link_href: str | None = None
        self._link_buf: list[str] = []
        self._row_cells: list[str] = []
        self._row_is_header = False
        self._cell_buf: list[str] = []
        self.meta_title: str = ""
        self.meta_description: str = ""
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.main_parts: list[str] = []
        self.article_parts: list[str] = []

    @staticmethod
    def _extract_code_lang(attrs: list[tuple[str, str | None]]) -> str:
        for k, v in attrs:
            if k.lower() == "class" and v:
                m = re.search(r"(?:language|lang)-([a-zA-Z0-9_+#.-]{1,20})", v)
                if m:
                    return m.group(1).lower()
        return ""

    def _emit(self, piece: str) -> None:
        if self._skip_depth > 0:
            return
        self.text_parts.append(piece)
        if self._main_depth > 0:
            self.main_parts.append(piece)
        if self._article_depth > 0:
            self.article_parts.append(piece)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "meta":
            amap = {k.lower(): (v or "").strip() for k, v in attrs}
            mname = (amap.get("name") or amap.get("property") or "").lower()
            mcontent = amap.get("content") or ""
            if mcontent:
                if mname in ("og:title", "twitter:title") and not self.meta_title:
                    self.meta_title = mcontent
                elif mname in ("description", "og:description", "twitter:description") and not self.meta_description:
                    self.meta_description = mcontent
            return
        if t == "svg":
            self._svg_depth += 1
        if t == "title" and self._svg_depth == 0 and not self._head_title_done:
            self._in_title = True
        if t == "header":
            should_skip = self._article_depth == 0 and self._main_depth == 0
            self._header_skip_stack.append(should_skip)
            if should_skip:
                self._skip_depth += 1
                return
            self._emit("\n")
            return
        if t in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if t in _MAIN_TAGS:
            self._main_depth += 1
        if t == "article":
            self._article_depth += 1

        if t in _HEADING_PREFIX:
            self._emit(f"\n\n{_HEADING_PREFIX[t]}")
        elif t == "li":
            self._emit("\n- ")
        elif t == "blockquote":
            self._emit("\n> ")
        elif t == "pre":
            self._in_pre = True
            lang = self._extract_code_lang(attrs)
            if lang:
                self._emit(f"\n\n```{lang}\n")
                self._pre_fence_emitted = True
            else:
                self._pre_fence_emitted = False
        elif t == "code":
            self._in_code = True
            if self._in_pre and not self._pre_fence_emitted:
                lang = self._extract_code_lang(attrs)
                self._emit(f"\n\n```{lang}\n")
                self._pre_fence_emitted = True
            elif not self._in_pre:
                self._emit("`")
        elif t == "a" and not self._in_pre and not self._in_cell:
            href = next((v.strip() for k, v in attrs if k.lower() == "href" and v), "")
            if href.startswith(("http://", "https://")) and len(href) <= 300:
                self._link_href = href
                self._link_buf = []
        elif t == "tr":
            self._row_cells = []
            self._row_is_header = False
        elif t in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []
            if t == "th":
                self._row_is_header = True
        elif t in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "svg" and self._svg_depth > 0:
            self._svg_depth -= 1
        if t == "title" and self._in_title:
            self._in_title = False
            self._head_title_done = True
        if t == "header":
            did_skip = self._header_skip_stack.pop() if self._header_skip_stack else True
            if did_skip:
                if self._skip_depth > 0:
                    self._skip_depth -= 1
                return
            self._emit("\n")
            return
        if t in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if t in _HEADING_PREFIX:
            self._emit("\n\n")
        elif t == "li":
            self._emit("\n")
        elif t == "pre":
            if not self._pre_fence_emitted:
                self._emit("\n\n```\n")
            self._in_pre = False
            self._pre_fence_emitted = False
            self._emit("\n```\n\n")
        elif t == "code":
            if not self._in_pre:
                self._emit("`")
            self._in_code = False
        elif t == "a" and self._link_href is not None:
            ltxt = re.sub(r"\s+", " ", "".join(self._link_buf)).strip()
            href = self._link_href
            self._link_href = None
            self._link_buf = []
            if ltxt and len(ltxt) <= 120 and ltxt != href:
                self._emit(f"[{ltxt}]({href})")
            elif ltxt:
                self._emit(ltxt)
        elif t in ("td", "th"):
            cell_txt = re.sub(r"\s+", " ", "".join(self._cell_buf)).strip().replace("|", "\\|")
            self._row_cells.append(cell_txt)
            self._in_cell = False
            self._cell_buf = []
        elif t == "tr":
            if self._row_cells:
                self._emit("\n| " + " | ".join(self._row_cells) + " |\n")
                if self._row_is_header:
                    self._emit("| " + " | ".join("---" for _ in self._row_cells) + " |\n")
            self._row_cells = []
            self._row_is_header = False
        elif t in _BLOCK_TAGS:
            self._emit("\n")

        if t in _MAIN_TAGS and self._main_depth > 0:
            self._main_depth -= 1
        if t == "article" and self._article_depth > 0:
            self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title and self._svg_depth == 0:
            self.title_parts.append(data)
        if self._skip_depth > 0:
            return
        if self._in_pre and not self._pre_fence_emitted and data.strip():
            self._emit("\n\n```\n")
            self._pre_fence_emitted = True
        if self._in_cell:
            self._cell_buf.append(data)
        elif self._link_href is not None:
            self._link_buf.append(data)
        else:
            self._emit(data)


_MIXED_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos|nbsp);", re.IGNORECASE)


def _unescape_html_entities(text: str) -> str:
    if "&" not in text:
        return text
    # 规范化大小写混合实体（如 &Amp;）并最多做两轮解码消除二次转义（&amp;amp;）
    cur = _MIXED_ENTITY_RE.sub(lambda m: f"&{m.group(1).lower()};", text)
    for _ in range(2):
        nxt = _html.unescape(cur)
        nxt = _MIXED_ENTITY_RE.sub(lambda m: f"&{m.group(1).lower()};", nxt)
        if nxt == cur:
            break
        cur = nxt
    return cur


def _normalize_extracted_text(raw: str) -> str:
    raw = _unescape_html_entities(raw)
    out: list[str] = []
    prev_empty = False
    in_fence = False
    for raw_ln in raw.splitlines():
        stripped = raw_ln.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(stripped)
            prev_empty = False
            continue
        if in_fence:
            code_ln = raw_ln.rstrip()
            if not code_ln:
                if not prev_empty and out:
                    out.append("")
                prev_empty = True
            else:
                out.append(code_ln)
                prev_empty = False
            continue
        ln = re.sub(r"[ \t]+", " ", raw_ln).strip()
        if not ln:
            if not prev_empty and out:
                out.append("")
            prev_empty = True
        else:
            out.append(ln)
            prev_empty = False
    return "\n".join(out).strip()


_PRUNE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "how", "what", "why", "are", "was", "were", "not", "can", "will",
}


def prune_content_for_query(body: str, query: str, max_chars: int) -> tuple[str, bool]:
    """查询聚焦段级裁剪：当长正文超出 max_chars 且给定 query 时，保留首段导语与最高相关性段落（按原顺序）。"""
    if not query.strip() or len(body) <= max_chars:
        return body[:max_chars], False

    raw_toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", query.lower())
    qterms: list[str] = []
    for tok in raw_toks:
        if tok in _PRUNE_STOPWORDS or len(tok) <= 1:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", tok) and len(tok) > 4:
            for i in range(len(tok) - 1):
                qterms.append(tok[i : i + 2])
        else:
            qterms.append(tok)
    qterms = list(dict.fromkeys(qterms))
    if not qterms:
        return body[:max_chars], False

    blocks = [b.strip() for b in re.split(r"\n{2,}", body) if b.strip()]
    if len(blocks) <= 1:
        return body[:max_chars], False

    # 为每个段落记录最近标题上下文与轻量 BM25-like 词项密度得分
    scored: list[tuple[float, int, str]] = []
    for idx, blk in enumerate(blocks):
        low = blk.lower()
        uniq_hits = sum(1 for t in qterms if t in low)
        tf_bonus = sum(min(3, low.count(t)) * 0.25 for t in qterms if t in low)
        len_norm = 1.0 + len(blk) / 800.0
        score = (uniq_hits * 2.0 + tf_bonus) / len_norm
        scored.append((score, idx, blk))

    if max(s for s, _, _ in scored) <= 0:
        return body[:max_chars], False

    selected_indices: set[int] = set()
    used_chars = 0

    # 先按相关性从高到低选入核心高分段落
    for score, idx, blk in sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True):
        if score <= 0:
            break
        piece_len = len(blk) + (2 if selected_indices else 0)
        if used_chars + piece_len <= max_chars:
            selected_indices.add(idx)
            used_chars += piece_len
        elif not selected_indices:
            # 单个最高分段落超长时截取该段
            return blk[:max_chars], True

    # 若还有余量，尽量保留第 0 段标题/导语提供上下文
    if 0 not in selected_indices:
        lead = blocks[0]
        if used_chars + len(lead) + 2 <= max_chars:
            selected_indices.add(0)
            used_chars += len(lead) + 2

    ordered = [blocks[i] for i in sorted(selected_indices)]
    pruned = "\n\n".join(ordered)
    if len(pruned) > max_chars:
        pruned = pruned[:max_chars]
    return pruned, True


__all__ = [
    "_BLOCK_TAGS",
    "_CleanTextExtractor",
    "_HEADING_PREFIX",
    "_MAIN_TAGS",
    "_MIXED_ENTITY_RE",
    "_PRUNE_STOPWORDS",
    "_SKIP_TAGS",
    "_normalize_extracted_text",
    "_unescape_html_entities",
    "prune_content_for_query",
]
