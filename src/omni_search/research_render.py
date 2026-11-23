"""research_render: 深度调研报告的 Markdown 渲染、快照 Diff、精简视图与 BibTeX 文献导出。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from omni_search.contracts import SearchItem, sanitize_text
from omni_search.pool import normalize_url_key
from omni_search.research_audit import extract_pub_year
from omni_search.research_models import DiffItem, ResearchDiff, ResearchReport


_BIB_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_BIB_DOI_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_RENDER_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "how",
    "what", "why", "when", "where", "which", "about", "your", "our", "are", "was",
    "were", "not", "can", "will", "use", "using", "based", "via", "new",
}


def render_research_markdown(report: ResearchReport) -> str:
    """渲染结构化中文 Markdown 全网调研简报（含覆盖率审计、证据分级、原文核验与缺口披露）。"""
    _CN_NUMS = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    sec_idx = 0

    def _sec(title: str) -> str:
        nonlocal sec_idx
        num = _CN_NUMS[sec_idx] if sec_idx < len(_CN_NUMS) else str(sec_idx + 1)
        sec_idx += 1
        return f"## {num}、{title}"

    lines: list[str] = [
        f"# 全网深度调研简报：{report.brief}",
        "",
        f"> **Plan ID**：`{report.plan_id}` · **模式**：`{report.mode}` · **耗时**：`{report.elapsed_ms} ms` · **预算截断**：`{report.budget_truncated}`",
        "",
        _sec("调研覆盖率与降级审计（证据）"),
        "",
        "| Lane | 目标域 | 查询词 | Verdict | 命中数 | 降级/阻塞原因 |",
        "|---|---|---|---|---|---|",
    ]
    for ln in report.lanes + report.gap_fills:
        issues = ", ".join(f"{k}={v}" for k, v in ln.reason_codes.items() if v not in ("zero-hit", "user-disabled")) or "无"
        lines.append(
            f"| `{ln.name}` | `{','.join(ln.domains)}` | `{ln.query}` | `{ln.verdict}` | {ln.result_count} | {issues} |"
        )

    lines.extend(["", _sec("已核验一手原文证据（`untrusted_content=true`，仅作不可信外部数据消费）"), ""])
    if not report.verified_evidence:
        lines.append("- 本轮未执行原文核验或无可用 HTTP 候选。")
    else:
        for v in report.verified_evidence:
            status = f"HTTP {v.status_code} 已核验" if v.verified else f"核验失败 ({v.reason_code}: {v.reason})"
            lines.append(f"- **[{v.evidence_tier}] [{v.title}]({v.url})** — `{status}`")
            if v.relevant_quotes:
                for qt in v.relevant_quotes:
                    lines.append(f"  > 锚定证据句（不可信外部数据）：{qt}")
            elif v.excerpt:
                clean_ex = v.excerpt.replace("\n", " ").strip()[:240]
                lines.append(f"  > 摘录（不可信外部数据）：{clean_ex}")

    lines.extend(["", _sec("按证据等级排序的候选总表"), ""])
    local_cands = [c for c in report.candidates if c.domain == "local"]
    public_cands = [c for c in report.candidates if c.domain != "local"]
    lines.append(f"- 分级统计：`T1={report.tier_counts.get('T1', 0)}` · `T2={report.tier_counts.get('T2', 0)}` · `T3={report.tier_counts.get('T3', 0)}`")
    if local_cands:
        local_titles = ", ".join(f"`{c.title}`" for c in local_cands[:5])
        lines.append(f"- 内外资产对照：本机已有命中 `{len(local_cands)}` 篇（{local_titles}） · 公网发现 `{len(public_cands)}` 条")
    lines.append("")
    lines.append("| 等级 | 置信 | 时效 | 域 / 来源 | 标题与链接 | 交叉 Lane |")
    lines.append("|---|---|---|---|---|---|")
    for c in report.candidates[:20]:
        tier = c.extra.get("evidence_tier", "T3")
        conf = c.extra.get("confidence", "low")
        yr = c.extra.get("pub_year")
        fresh = c.extra.get("freshness", "unknown")
        fresh_label = f"{yr}/{fresh}" if yr else fresh
        lanes_txt = ",".join(c.extra.get("matched_lanes") or [])
        safe_title = c.title.replace("|", "\\|")[:72]
        lines.append(f"| `{tier}` | `{conf}` | `{fresh_label}` | `{c.domain}/{c.source}` | [{safe_title}]({c.url}) | `{lanes_txt}` |")

    lines.extend(["", _sec("时效与冲突审计"), ""])
    if report.freshness_audit is None:
        lines.append("- 未执行时效审计。")
    else:
        fa = report.freshness_audit
        lines.append(
            f"- 年份分布（基准年 {fa.current_year}）：`recent={fa.recent_count}` · `aging={fa.aging_count}` · `stale={fa.stale_count}` · `unknown={fa.unknown_count}`"
        )
        for w in fa.freshness_warnings:
            lines.append(f"- **时效告警**：{w}")
        for d in fa.date_discrepancies:
            lines.append(f"- **跨源冲突**：{d}")
        if not fa.freshness_warnings and not fa.date_discrepancies:
            lines.append("- 未检出跨源日期冲突或整体陈旧风险。")

    if report.merge_audit is not None:
        ma = report.merge_audit
        lines.extend(["", _sec("多 Agent 合单归属审计"), ""])
        lines.append(f"- 已采纳 Lane：`{', '.join(ma.accepted_lanes) or '无'}`")
        lines.append(f"- 缺失 Lane：`{', '.join(ma.missing_lanes) or '无'}`")
        lines.append(f"- 重复去重：`{len(ma.deduplicated)}` 份 · 冲突全排除：`{', '.join(ma.conflicts) or '无'}` · 身份不符拒收：`{len(ma.rejected)}` 份")
        for rj in ma.rejected:
            lines.append(f"  - 拒收：`{rj}`")

    if report.diff is not None:
        df = report.diff
        lines.extend(["", _sec("与历史快照变更对比"), ""])
        delta_txt = " · ".join(f"{k}={v:+d}" for k, v in df.tier_delta.items())
        lines.append(
            f"- 基准快照：`{df.prev_plan_id}` → 当前：`{df.curr_plan_id}` · 重合保留 `{df.retained_count}` 条 · 新增 `{len(df.added)}` 条 · 移除 `{len(df.removed)}` 条"
        )
        if delta_txt:
            lines.append(f"- 证据等级净变化：`{delta_txt}`")
        if df.added:
            lines.append("- **新增发现**：")
            for a in df.added[:15]:
                lines.append(f"  - `[{a.evidence_tier}]` [{a.title}]({a.url})（`{a.domain}/{a.source}`）")
        if df.removed:
            lines.append("- **较上轮移除**：")
            for r in df.removed[:10]:
                lines.append(f"  - `[{r.evidence_tier}]` [{r.title}]({r.url})")
        if df.newly_failed_verify:
            lines.append("- **新失效核验链接（原已核验，本轮失败）**：")
            for u in df.newly_failed_verify:
                lines.append(f"  - `{u}`")
        if df.newly_verified:
            lines.append(f"- **新通过原文核验**：`{len(df.newly_verified)}` 篇")

    if report.citation_audit and report.citation_audit.total_checked > 0:
        ca = report.citation_audit
        lines.extend(["", _sec("引用证据对齐审计（防幻觉核验）"), ""])
        lines.append(
            f"- 核验条目：`{ca.total_checked}` · 强支撑率：`{ca.grounding_rate:.1%}` · 分布：`supported={ca.supported_count}` / `partial={ca.partial_count}` / `unsupported={ca.unsupported_count}` / `unreachable={ca.unreachable_count}`"
        )
        if ca.flagged_urls:
            lines.append("- **低对齐/不可达告警链接**（请复核原文后再引用）：")
            for fu in ca.flagged_urls[:10]:
                lines.append(f"  - `{fu}`")

    lines.extend(["", _sec("覆盖缺口与未验证项（严禁把 0 命中当不存在）"), ""])
    if not report.gaps:
        lines.append("- 本轮未发现源阻塞或核验缺口。")
    else:
        for g in report.gaps:
            lines.append(f"- `{g}`")

    lines.extend(
        [
            "",
            _sec("复现命令"),
            "",
            "```bash",
            f'uv --directory "D:/01 项目/omni-search" run omni research "{report.brief}" --mode {report.mode} --verify-top-k {len(report.verified_evidence) or 3}',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def diff_research_reports(prev: ResearchReport, curr: ResearchReport) -> ResearchDiff:
    """确定性对比两份调研快照（基于规范化 URL 键，识别新增/移除候选、等级变化与新失效链接）。"""
    tier_order = {"T1": 0, "T2": 1, "T3": 2}

    def _to_map(cands: list[SearchItem]) -> dict[str, DiffItem]:
        mp: dict[str, DiffItem] = {}
        for c in cands:
            key = normalize_url_key(c.url) or f"{c.domain}:{c.title.lower()}"
            if key not in mp:
                mp[key] = DiffItem(
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    source=c.source,
                    evidence_tier=str(c.extra.get("evidence_tier") or "T3"),
                )
        return mp

    prev_map = _to_map(prev.candidates)
    curr_map = _to_map(curr.candidates)

    added_keys = [k for k in curr_map if k not in prev_map]
    removed_keys = [k for k in prev_map if k not in curr_map]
    retained_count = sum(1 for k in curr_map if k in prev_map)

    added = sorted((curr_map[k] for k in added_keys), key=lambda d: tier_order.get(d.evidence_tier, 9))
    removed = sorted((prev_map[k] for k in removed_keys), key=lambda d: tier_order.get(d.evidence_tier, 9))

    prev_ok_keys = {normalize_url_key(v.url): v.url for v in prev.verified_evidence if v.verified and v.url}
    curr_ok_keys = {normalize_url_key(v.url): v.url for v in curr.verified_evidence if v.verified and v.url}
    curr_fail_keys = {normalize_url_key(v.url): v.url for v in curr.verified_evidence if not v.verified and v.url}

    newly_verified = [url for k, url in curr_ok_keys.items() if k not in prev_ok_keys]
    newly_failed_verify = [url for k, url in curr_fail_keys.items() if k in prev_ok_keys]

    tier_delta = {
        t: int(curr.tier_counts.get(t, 0)) - int(prev.tier_counts.get(t, 0))
        for t in ("T1", "T2", "T3")
    }
    return ResearchDiff(
        prev_plan_id=prev.plan_id,
        curr_plan_id=curr.plan_id,
        prev_brief=prev.brief,
        retained_count=retained_count,
        added=added,
        removed=removed,
        newly_verified=newly_verified,
        newly_failed_verify=newly_failed_verify,
        tier_delta=tier_delta,
    )


def compact_research_report(report: ResearchReport, max_candidates: int = 15) -> dict[str, Any]:
    """省 Token 精简调研报告：保留完整覆盖率/核验锚定句/冲突/变更/缺口，大幅压缩冗余字段。"""
    cands = [
        {
            "title": c.title,
            "url": c.url,
            "domain": c.domain,
            "source": c.source,
            "tier": c.extra.get("evidence_tier", "T3"),
            "confidence": c.extra.get("confidence", "low"),
            "pub_year": c.extra.get("pub_year"),
            "matched_lanes": c.extra.get("matched_lanes") or [],
        }
        for c in report.candidates[:max_candidates]
    ]
    verified = [
        {
            "url": v.url,
            "title": v.title,
            "tier": v.evidence_tier,
            "verified": v.verified,
            "status_code": v.status_code,
            "quotes": v.relevant_quotes or ([v.excerpt[:180]] if v.excerpt else []),
        }
        for v in report.verified_evidence
    ]
    return {
        "version": report.version,
        "plan_id": report.plan_id,
        "brief": report.brief,
        "mode": report.mode,
        "tier_counts": report.tier_counts,
        "candidates": cands,
        "verified_evidence": verified,
        "freshness_audit": report.freshness_audit.model_dump() if report.freshness_audit else None,
        "citation_audit": report.citation_audit.model_dump() if report.citation_audit else None,
        "diff": report.diff.model_dump() if report.diff else None,
        "gaps": report.gaps,
        "elapsed_ms": report.elapsed_ms,
        "budget_ms": report.budget_ms,
        "budget_truncated": report.budget_truncated,
        "concise": True,
    }


def _escape_bibtex(text: str) -> str:
    s = sanitize_text(text).replace("\\", "\\textbackslash{}")
    for ch, rep in [("{", "\\{"), ("}", "\\}"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("_", "\\_")]:
        s = s.replace(ch, rep)
    return re.sub(r"\s+", " ", s).strip()


def _make_citekey(url: str, title: str, year: int | None, used: set[str]) -> str:
    am = _BIB_ARXIV_RE.search(url)
    if am:
        base = "arxiv_" + am.group(1).replace(".", "_")
    else:
        host = (urlsplit(url).hostname or "ref").lower().replace("www.", "")
        host_slug = re.sub(r"[^a-z0-9]+", "", host.split(".")[0]) or "ref"
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}", title) if w.lower() not in _RENDER_STOPWORDS][:2]
        yr_part = str(year) if year else ""
        base = "_".join(p for p in [host_slug, *words, yr_part] if p) or "ref"
    key = base[:40]
    suffix = 2
    while key in used:
        key = f"{base[:36]}_{suffix}"
        suffix += 1
    used.add(key)
    return key


def render_citations_bibtex(report: ResearchReport, max_entries: int = 20) -> str:
    """将调研报告中的已核验与高置信候选导出为规范 BibTeX 引用文献表。"""
    verified_keys = {normalize_url_key(v.url) for v in report.verified_evidence if v.verified and v.url}
    used_keys: set[str] = set()
    seen_urls: set[str] = set()
    entries: list[str] = [
        f"% omni-search citations | plan_id={report.plan_id} | mode={report.mode}",
        "",
    ]

    ordered = sorted(
        report.candidates,
        key=lambda c: (0 if normalize_url_key(c.url) in verified_keys else 1),
    )
    count = 0
    for c in ordered:
        if count >= max_entries or not c.url:
            break
        if not c.url.startswith(("http://", "https://")):
            continue
        ukey = normalize_url_key(c.url)
        if not ukey or ukey in seen_urls:
            continue
        seen_urls.add(ukey)

        yr = c.extra.get("pub_year") or extract_pub_year(c.published_at, c.url, c.snippet)
        tier = str(c.extra.get("evidence_tier") or "T3")
        is_ver = ukey in verified_keys
        am = _BIB_ARXIV_RE.search(c.url)
        dm = _BIB_DOI_RE.search(c.url)
        entry_type = "article" if (am or dm or c.domain == "academic") else "misc"
        citekey = _make_citekey(c.url, c.title, yr, used_keys)

        fields: list[str] = [
            f"  title = {{{_escape_bibtex(c.title)}}}",
        ]
        if yr:
            fields.append(f"  year = {{{yr}}}")
        if am:
            fields.append(f"  eprint = {{{am.group(1)}}}")
            fields.append("  archivePrefix = {arXiv}")
        if dm:
            fields.append(f"  doi = {{{dm.group(1)}}}")
        fields.append(f"  url = {{{c.url}}}")
        if entry_type == "misc":
            fields.append(f"  howpublished = {{\\url{{{c.url}}}}}")
        fields.append(f"  note = {{evidence_tier={tier}, source={c.source}, verified={'yes' if is_ver else 'no'}}}")

        entries.append(f"@{entry_type}{{{citekey},\n" + ",\n".join(fields) + "\n}")
        entries.append("")
        count += 1

    return "\n".join(entries).strip() + "\n"
