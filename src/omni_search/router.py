"""规则优先的零延迟意图域路由器。"""

from __future__ import annotations

import re

VALID_DOMAINS = (
    "web",
    "news",
    "knowledge",
    "academic",
    "code",
    "code_evidence",
    "packages",
    "security",
    "media",
    "local",
    "local_code",
)

_LOCAL_RE = re.compile(
    r"\b(local|my files|my notes|desktop|folder|workspace|pdf|screenshot|screenshots)\b|本地|本机|我的文件|笔记|截图|磁盘|目录",
    re.IGNORECASE,
)
_LOCAL_CODE_RE = re.compile(
    r"\b(local code|workspace code|local repo|local source|ripgrep|ast-grep)\b|本地代码|本机代码|本地源码|本机源码|当前仓库|本地仓库",
    re.IGNORECASE,
)
_LOCAL_KB_FILTER_RE = re.compile(
    r"(?:^|\s)(?:type|ext|after|before):[^\s]+",
    re.IGNORECASE,
)
_LOCAL_CODE_FILTER_RE = re.compile(
    r"(?:^|\s)(?:symbol|ast|lang):[^\s]+",
    re.IGNORECASE,
)
_SHARED_PATH_FILTER_RE = re.compile(
    r"(?:^|\s)-?path:[^\s]+",
    re.IGNORECASE,
)
_CODE_EVIDENCE_RE = re.compile(
    r"\b(code evidence|usage example|implementation|snippet|grep|sourcegraph|traceback|stacktrace|how to call)\b|代码实现|代码片段|源码片段|调用示例|怎么调用|实现示例",
    re.IGNORECASE,
)
_SECURITY_RE = re.compile(
    r"\b(cve-\d{4}-\d+|cve|nvd|vulnerability|vulnerabilities|exploit|advisory|owasp|ssrf|xss|rce|sqli|ghsa|zero-day|0day)\b|漏洞|安全公告|提权|注入攻击|渗透",
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"\b(github|gitlab|gitee|repo|repository|library|sdk|crate|npm|pypi|nuget|maven|rust|python|typescript|golang|docker|mcp|code|api|cli|framework)\b|开源|仓库|代码|框架|源码|包|依赖",
    re.IGNORECASE,
)
_ACADEMIC_RE = re.compile(
    r"\b(arxiv|paper|papers|doi|citation|survey|benchmark|transformer|llm|multimodal|retrieval|algorithm|study|journal|thesis|preprint)\b|论文|研究|综述|学术|专利|算法|实验",
    re.IGNORECASE,
)
_NEWS_RE = re.compile(
    r"\b(news|today|latest|breaking|release|launch|announced|202[5-9]|hn|hackernews|trend|update|event|changelog|roadmap)\b|新闻|最新|今日|发布|动态|热点|进展|新版本",
    re.IGNORECASE,
)
_MEDIA_RE = re.compile(
    r"\b(dataset|model|weights|huggingface|archive|book|movie|video|audio|image|photo|media|podcast)\b|数据集|模型|图书|视频|音频|图片|档案馆",
    re.IGNORECASE,
)
_KNOWLEDGE_RE = re.compile(
    r"\b(what is|who is|define|definition|history|wiki|wikipedia|city|country|location|geography|concept|standard|rfc)\b|是什么|定义|简介|百科|历史|概念|标准|在哪",
    re.IGNORECASE,
)


def route_domains(query: str, requested_domains: list[str] | None = None) -> list[str]:
    """把用户输入路由到目标域；显式指定域时尊重用户，auto 时按规则组合多域。"""
    if requested_domains:
        cleaned = [d.strip().lower() for d in requested_domains if d and d.strip()]
        if cleaned and "all" in cleaned:
            return list(VALID_DOMAINS)
        if cleaned and "auto" not in cleaned:
            valid = [d for d in cleaned if d in VALID_DOMAINS]
            if valid:
                return list(dict.fromkeys(valid))

    selected: list[str] = ["web", "knowledge"]
    if _CODE_RE.search(query):
        selected.extend(["code", "packages"])
    if _CODE_EVIDENCE_RE.search(query):
        selected.append("code_evidence")
    if _SECURITY_RE.search(query):
        selected.append("security")
    if _ACADEMIC_RE.search(query):
        selected.append("academic")
    if _NEWS_RE.search(query):
        selected.append("news")
    if _MEDIA_RE.search(query):
        selected.append("media")
    if _LOCAL_RE.search(query):
        selected.append("local")
    if _LOCAL_CODE_RE.search(query):
        selected.extend(["local_code", "local"])
    if _SHARED_PATH_FILTER_RE.search(query):
        selected.extend(["local", "local_code"])
    if _LOCAL_CODE_FILTER_RE.search(query):
        selected.append("local_code")
    if _LOCAL_KB_FILTER_RE.search(query):
        selected.append("local")
    if _KNOWLEDGE_RE.search(query):
        selected.append("knowledge")

    # 若没有任何专项命中，默认带 news + academic 扩充公网万物召回面
    if len(selected) == 2:
        selected.extend(["news", "academic"])

    return list(dict.fromkeys(selected))
