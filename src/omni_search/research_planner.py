"""omni_research 规划与词元分析纯函数（多模式 Lane 规划、信息量加权宽词提取与确定性自反追问提词）。"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field

from omni_search.contracts import SearchItem

VALID_RESEARCH_MODES = ("standard", "feasibility", "benchmark", "factcheck")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "on", "to", "with", "by", "from",
    "how", "what", "why", "when", "where", "is", "are", "be", "can", "do", "does",
    "build", "design", "make", "using", "use", "about", "best", "guide", "tutorial",
}


class ResearchLane(BaseModel):
    name: str
    query: str
    domains: list[str]
    purpose: str = ""


class ResearchPlan(BaseModel):
    plan_id: str
    brief: str
    mode: str = "standard"
    extra_terms: list[str] = Field(default_factory=list)
    lanes: list[ResearchLane]


_ZH_SPLIT_RE = re.compile(r"(?:关于|如何|怎样|怎么|什么|基于|通过|以及|并且|或者|进行|实现|用于|[的与和及是])+")
_ZH_EDGE_STOP = set("的了和与在中是对为及等将把被从到之其于或以所能可不有无上下内外这那该本文做用")


def _is_cjk_dominant(text: str) -> bool:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return cjk >= 2 and cjk >= latin


_GENERIC_BROAD_TERMS = {
    "ai", "web", "search", "server", "local", "hybrid", "open", "source", "system", "systems",
    "tool", "tools", "framework", "library", "engine", "data", "model", "models", "agent",
    "agents", "app", "application", "service", "platform", "code", "file", "files", "document",
    "documents", "based", "using", "build", "best", "modern", "fast", "simple", "guide",
    "tutorial", "overview", "study", "analysis", "method", "approach", "architecture", "design",
    "review", "survey", "comparison", "benchmark", "evaluation", "parser", "vector",
    "research", "deep", "learning", "performance", "workflow", "pipeline",
}


def _token_specificity_score(tok: str) -> float:
    """按词项信息区分度打分（技术专名/缩写/含数字连字符 > 普通实词 > 高频泛词），用于智能提词与相关性加权。"""
    low = tok.lower()
    if low in _GENERIC_BROAD_TERMS:
        return -2.5
    score = 0.0
    if any(ch in tok for ch in ("-", "_", ".", "/")):
        score += 4.0
    if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
        score += 3.5
    # 全大写或驼峰技术缩写/产品名（如 MCP, RAG, FTS5, MarkItDown）
    if (tok.isupper() and 2 <= len(tok) <= 8) or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok)):
        score += 3.0
    if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
        score += 2.0 if len(tok) >= 3 else 1.0
    elif len(low) >= 7:
        score += 2.0
    elif len(low) >= 5:
        score += 1.5
    else:
        score += 0.5
    return score


def widen_query(brief: str, max_terms: int = 3) -> str:
    """从长自然语言 brief 中提取高区分度核心技术词（支持无空格中文切分与信息量加权，避免截断丢核心实体）。"""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", brief)
    expanded: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", tok) and len(tok) > 6:
            parts = [p for p in _ZH_SPLIT_RE.split(tok) if len(p) >= 2]
            if len(parts) >= 2:
                expanded.extend(parts)
            elif len(tok) > 8:
                expanded.append(tok[:6])
                expanded.append(tok[-4:])
            else:
                expanded.append(tok)
        else:
            expanded.append(tok)

    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for tok in expanded:
        low = tok.lower()
        if low in _STOPWORDS or low.isdigit() or len(low) <= 1:
            continue
        if low in seen:
            continue
        seen.add(low)
        candidates.append((len(candidates), tok))

    if not candidates:
        return brief.strip()[:64]
    if len(candidates) <= max_terms:
        return " ".join(tok for _, tok in candidates)

    ranked = sorted(candidates, key=lambda pair: (_token_specificity_score(pair[1]), -pair[0]), reverse=True)
    chosen = sorted(ranked[:max_terms], key=lambda pair: pair[0])
    return " ".join(tok for _, tok in chosen)


def _compute_query_relevance(item: SearchItem, brief: str) -> tuple[float, int]:
    """计算候选条目对 brief 锚词的加权覆盖率（高区分度技术词权重 2.0，泛词 0.7），返回 (score 0..1, 锚词数)。"""
    if not brief.strip():
        return 1.0, 0
    raw_toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}", brief)
    anchors: list[tuple[str, float]] = []
    seen: set[str] = set()
    for tok in raw_toks:
        low = tok.lower()
        if low in _STOPWORDS or low.isdigit() or len(low) <= 1:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", tok) and len(tok) > 4:
            for i in range(len(tok) - 1):
                g = tok[i : i + 2]
                if g[0] in _ZH_EDGE_STOP or g[-1] in _ZH_EDGE_STOP or g in seen:
                    continue
                seen.add(g)
                anchors.append((g, 1.5))
            continue
        if low in seen:
            continue
        seen.add(low)
        w = 0.7 if low in _GENERIC_BROAD_TERMS else (2.0 if _token_specificity_score(tok) >= 1.5 else 1.2)
        anchors.append((low, w))

    if not anchors:
        return 1.0, 0
    title_low = (item.title or "").lower()
    snip_low = (item.snippet or "").lower()
    total_w = sum(w for _, w in anchors)
    hit_w = 0.0
    for term, w in anchors:
        if term in title_low:
            hit_w += w
        elif term in snip_low:
            hit_w += w * 0.55
    return min(1.0, hit_w / total_w) if total_w > 0 else 1.0, len(anchors)


def plan_research_lanes(
    brief: str,
    domains: list[str] | None = None,
    mode: str = "standard",
    extra_terms: list[str] | None = None,
    include_local: bool = False,
) -> ResearchPlan:
    """按调研模式确定性拆分差异化多视角 lane，并绑定唯一 plan_id。"""
    eff_mode = mode.strip().lower() if mode else "standard"
    if eff_mode not in VALID_RESEARCH_MODES:
        eff_mode = "standard"

    clean_brief = " ".join(brief.strip().split())
    terms = [t.strip() for t in (extra_terms or []) if t and t.strip()]
    core = widen_query(clean_brief, max_terms=4)
    term_suffix = (" " + " ".join(terms[:3])) if terms else ""

    allowed = {d.strip().lower() for d in domains if d and d.strip()} if domains else None
    if allowed and ("auto" in allowed or "all" in allowed):
        allowed = None

    is_zh = _is_cjk_dominant(clean_brief)
    if eff_mode == "feasibility":
        raw_lanes = [
            ("overview", f"{clean_brief}{term_suffix}".strip(), ["web", "knowledge"], "全景现状与概念边界"),
            (
                "alternatives",
                f"{core}{term_suffix} {'开源 方案' if is_zh else 'open source alternative'}".strip(),
                ["web", "code", "packages"],
                "现有开源标杆与竞品",
            ),
            (
                "academic",
                f"{core}{term_suffix} {'架构 综述' if is_zh else 'architecture survey'}".strip(),
                ["academic"],
                "学术基础与前沿架构",
            ),
            (
                "risks",
                f"{core}{term_suffix} {'局限 挑战' if is_zh else 'limitations challenges'}".strip(),
                ["web", "news", "knowledge"],
                "工程瓶颈、失败案例与风险",
            ),
            ("community", f"{core}{term_suffix}".strip(), ["news", "media"], "近期动态与生态资源"),
        ]
    elif eff_mode == "benchmark":
        raw_lanes = [
            (
                "alternatives",
                f"{core}{term_suffix} {'开源' if is_zh else 'open source'}".strip(),
                ["code", "packages"],
                "候选开源项目发现",
            ),
            (
                "comparison",
                f"{core}{term_suffix} {'对比 评测' if is_zh else 'comparison benchmark'}".strip(),
                ["web", "news"],
                "横向对比与性能实测",
            ),
            (
                "academic",
                f"{core}{term_suffix} {'论文 评测' if is_zh else 'evaluation'}".strip(),
                ["academic"],
                "论文基准与评测方法",
            ),
            ("overview", f"{clean_brief}{term_suffix}".strip(), ["knowledge", "media"], "术语定义与生态资料"),
        ]
    elif eff_mode == "factcheck":
        raw_lanes = [
            (
                "authoritative",
                f"{clean_brief}{term_suffix} {'官方 规范' if is_zh else 'official'}".strip(),
                ["web", "academic"],
                "一手官方与论文溯源",
            ),
            ("encyclopedia", f"{core}{term_suffix}".strip(), ["knowledge"], "百科与实体知识核对"),
            ("news_verify", f"{clean_brief}{term_suffix}".strip(), ["news"], "时事与社区交叉验证"),
        ]
    else:
        raw_lanes = [
            ("overview", f"{clean_brief}{term_suffix}".strip(), ["web", "knowledge"], "通用全景与百科"),
            ("academic", f"{clean_brief}{term_suffix}".strip(), ["academic"], "学术论文与文献"),
            ("community", f"{core}{term_suffix}".strip(), ["news"], "社区讨论与最新资讯"),
            ("implementation", f"{core}{term_suffix}".strip(), ["code", "packages", "media"], "开源实现、包与数字资源"),
        ]

    lanes: list[ResearchLane] = []
    for name, q, doms, purpose in raw_lanes:
        eff_doms = [d for d in doms if allowed is None or d in allowed]
        if eff_doms:
            lanes.append(ResearchLane(name=name, query=q, domains=eff_doms, purpose=purpose))

    if not lanes:
        fallback_doms = sorted(allowed) if allowed else ["web", "knowledge", "academic", "news"]
        lanes.append(ResearchLane(name="overview", query=f"{clean_brief}{term_suffix}".strip(), domains=fallback_doms, purpose="保底检索"))

    if include_local and not any("local" in ln.domains for ln in lanes):
        lanes.insert(
            0,
            ResearchLane(
                name="local_kb",
                query=f"{clean_brief}{term_suffix}".strip(),
                domains=["local"],
                purpose="本机知识库与代码已有资产",
            ),
        )

    sig = f"{eff_mode}|{clean_brief}|{terms}|{include_local}|{[(l.name, l.query, l.domains) for l in lanes]}"
    plan_id = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
    return ResearchPlan(plan_id=plan_id, brief=clean_brief, mode=eff_mode, extra_terms=terms, lanes=lanes)


_DEEPEN_IGNORE = _STOPWORDS | {
    "open", "source", "paper", "papers", "study", "work", "works", "this", "that", "these", "those",
    "with", "from", "into", "over", "under", "between", "through", "such", "also", "more", "most",
    "very", "only", "used", "uses", "based", "system", "systems", "method", "methods", "approach",
    "framework", "library", "tool", "tools", "extension", "engine", "data", "model", "models",
    "evaluation", "benchmark", "comparison", "survey", "alternative", "official", "limitations",
    "challenges", "local", "article", "blog", "post", "github", "arxiv", "http", "https", "www",
    "com", "org", "note", "notes", "file", "files", "result", "results", "combines", "compares",
    "published", "updated", "version", "release", "readme", "documentation", "plugin", "plugins",
    "store", "awesome", "review", "analysis", "introduction", "overview", "project", "projects",
    "your", "our", "their", "its", "you", "they", "them", "new", "all", "any", "one", "two",
    "via", "per", "without", "just", "like", "top", "free", "fast", "simple", "easy", "modern",
    "full", "complete", "docs", "doc", "api", "apis", "cli", "sdk", "app", "apps",
    "open-source", "real-world", "state-of-the-art", "end-to-end", "built-in", "high-level",
    "long-term", "short-term", "well-known", "multi-agent", "non-trivial",
    # 中文通用泛词
    "基于", "通过", "使用", "实现", "研究", "方法", "系统", "技术", "模型", "分析", "应用",
    "介绍", "指南", "开源", "方案", "以及", "可以", "一种", "相关", "问题", "设计", "实践",
    "框架", "优化", "提升", "本文", "结合", "融合", "进行", "采用", "提供", "支持", "架构",
    "面向", "体系", "构建", "评估", "综合", "平台", "能力", "解析", "领域", "场景", "标准",
    "路径", "关键", "驱动", "趋势", "进展", "综述", "模式", "机制", "策略", "效果", "性能",
}
_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+|[A-Za-z][A-Za-z0-9_]{2,}")
_ZH_SEG_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_ZH_CLAUSE_SPLIT_RE = re.compile(r"[的了和与在中是对为及或将把被从到之其于以等并]+")


def _extract_candidate_terms(text: str, brief_low: str, brief_toks: set[str]) -> set[str]:
    terms: set[str] = set()
    for m in _ASCII_TERM_RE.finditer(text):
        tok = m.group(0).lower().strip("-_")
        if len(tok) < 3 or tok.isdigit() or tok in _DEEPEN_IGNORE or tok in brief_toks or tok in brief_low:
            continue
        terms.add(tok)
    for m in _ZH_SEG_RE.finditer(text):
        for seg in _ZH_CLAUSE_SPLIT_RE.split(m.group(0)):
            if len(seg) < 2:
                continue
            for n in (4, 3, 2):
                if len(seg) < n:
                    continue
                for i in range(len(seg) - n + 1):
                    gram = seg[i : i + n]
                    if gram[0] in _ZH_EDGE_STOP or gram[-1] in _ZH_EDGE_STOP:
                        continue
                    if gram in _DEEPEN_IGNORE or gram in brief_toks or gram in brief_low:
                        continue
                    # 4 字中文若前后半段含泛词（如 "场景构建"/"体系构建"），拒绝拼装伪术语
                    if n == 4 and (gram[:2] in _DEEPEN_IGNORE or gram[2:] in _DEEPEN_IGNORE):
                        continue
                    terms.add(gram)
    return terms


def extract_followup_terms(
    candidates: list[SearchItem],
    brief: str,
    max_terms: int = 2,
) -> list[str]:
    """确定性自反提词：从首轮多条候选交叉共振（df>=2）的标题/摘要中提取中英文高信息量新术语。"""
    brief_low = brief.lower()
    brief_toks = set(re.findall(r"[a-z0-9-]+|[\u4e00-\u9fff]{2,}", brief_low))
    for bt in list(brief_toks):
        if "-" in bt:
            brief_toks.update(bt.split("-"))

    doc_freq: dict[str, int] = {}
    for c in candidates[:15]:
        text = f"{c.title} {c.snippet}"
        for tok in _extract_candidate_terms(text, brief_low, brief_toks):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    resonant = [(tok, df) for tok, df in doc_freq.items() if df >= 2]
    if not resonant:
        return []

    is_zh_brief = _is_cjk_dominant(brief)

    def _rank_key(pair: tuple[str, int]) -> tuple[int, int, int, int, str]:
        tok, df = pair
        is_cjk = 1 if re.fullmatch(r"[\u4e00-\u9fff]+", tok) else 0
        # 中文主题适度偏好 3-4 字中文专有名词，英文主题偏好含连字符/数字的技术词
        lang_bonus = 1 if (is_zh_brief and is_cjk) else 0
        is_tech = 1 if ("-" in tok or any(ch.isdigit() for ch in tok) or (is_cjk and len(tok) >= 3)) else 0
        return (df, lang_bonus, is_tech, min(len(tok), 18), tok)

    resonant.sort(key=_rank_key, reverse=True)
    picked: list[str] = []
    for tok, _ in resonant:
        # 子串剪枝：避免同时选中互为包含的重复词（如 "知识图谱" 与 "知识"、"sqlite-vec" 与 "vec"）
        if any(tok in p or p in tok for p in picked):
            continue
        picked.append(tok)
        if len(picked) >= max_terms:
            break
    return picked
