"""omni CLI 入口：search / fetch / batch / doctor / mcp。"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from typing import Any

from omni_search import __version__
from omni_search.contracts import compact_search_response
from omni_search.fetcher import safe_fetch_url, safe_fetch_urls_batch
from omni_search.local_index import index_paths
from omni_search.mcp_server import OmniMCPServer, PROJECT_ROOT
from omni_search.monitor import render_monitor_markdown, run_monitor_check
from omni_search.site_map import safe_crawl_site, safe_map_site
from omni_search.verifier import ClaimInput, render_verification_markdown, verify_claims
from omni_search.research import (
    ResearchReport,
    compact_research_report,
    diff_research_reports,
    export_research_workorders,
    merge_lane_results,
    plan_research_lanes,
    render_citations_bibtex,
    render_research_markdown,
    run_research_rounds,
)


def _emit(payload: Any, compact: bool = False, out_path: str | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=None if compact else 2)
    if out_path:
        p = pathlib.Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        print(f"[omni] written {len(text.encode('utf-8'))} bytes -> {p}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        print(text)


async def _run(args: argparse.Namespace) -> int:
    server = OmniMCPServer()
    try:
        return await _run_with_server(server, args)
    finally:
        await server.aclose()


async def _run_with_server(server: OmniMCPServer, args: argparse.Namespace) -> int:
    if args.cmd == "mcp":
        if getattr(args, "transport", "stdio") == "http":
            import os
            from omni_search.mcp_http import McpHttpServer

            token = str(getattr(args, "token", "") or os.environ.get("OMNI_MCP_TOKEN", "") or "").strip()
            http_srv = McpHttpServer(
                server,
                host=str(getattr(args, "host", "127.0.0.1") or "127.0.0.1"),
                port=int(getattr(args, "port", 8765)),
                auth_token=token,
            )
            print(
                f"[omni] MCP Streamable HTTP listening on http://{http_srv.host}:{http_srv.requested_port}/mcp (auth={'bearer' if token else 'loopback-only'})",
                file=sys.stderr,
            )
            try:
                await http_srv.serve_forever()
            finally:
                await server.aclose()
            return 0
        try:
            await server.serve_stdio()
        finally:
            await server.aclose()
        return 0
    if args.cmd == "doctor":
        raw_re = str(getattr(args, "reset_engines", "") or "").strip()
        re_list = [x.strip() for x in raw_re.split(",") if x.strip()] if raw_re else None
        payload = await server.doctor(
            probe=args.probe,
            reset_health=getattr(args, "reset_health", False),
            reset_engines=re_list,
            stats=bool(getattr(args, "stats", False)),
            stats_days=int(getattr(args, "stats_days", 7)),
        )
        _emit(payload, compact=args.compact, out_path=args.out)
        return 0
    if args.cmd == "search":
        domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else ["auto"]
        resp = await server.pool.search(
            query=args.query,
            domains=domains,
            limit=args.limit,
            max_results=args.max_results,
            budget_ms=args.budget_ms,
            use_cache=not getattr(args, "no_cache", False),
            auto_relax=not getattr(args, "no_relax", False),
        )
        out_payload = compact_search_response(resp) if getattr(args, "concise", False) else resp.model_dump()
        _emit(out_payload, compact=args.compact, out_path=args.out)
        return 0 if resp.coverage.executed else 4
    if args.cmd == "fetch":
        urls = [str(u) for u in (getattr(args, "urls", None) or []) if str(u).strip()]
        if len(urls) <= 1:
            res = await safe_fetch_url(
                url=urls[0] if urls else "",
                max_chars=args.max_chars,
                timeout_sec=args.timeout,
                disk_cache=server.disk_cache,
                use_cache=not getattr(args, "no_cache", False),
                prefer_llms_txt=bool(getattr(args, "llms_txt", False)),
                query=str(getattr(args, "query", "") or ""),
            )
            _emit(res.model_dump(), compact=args.compact, out_path=args.out)
            return 0 if res.ok else 1
        bres = await safe_fetch_urls_batch(
            urls=urls,
            max_chars=args.max_chars,
            timeout_sec=args.timeout,
            concurrency=int(getattr(args, "concurrency", 5)),
            disk_cache=server.disk_cache,
            use_cache=not getattr(args, "no_cache", False),
            prefer_llms_txt=bool(getattr(args, "llms_txt", False)),
            query=str(getattr(args, "query", "") or ""),
        )
        _emit(bres.model_dump(), compact=args.compact, out_path=args.out)
        return 0 if bres.ok else 1
    if args.cmd == "map":
        mres = await safe_map_site(
            url=args.url,
            query=str(getattr(args, "query", "") or ""),
            max_urls=int(args.max_urls),
            timeout_sec=float(args.timeout),
        )
        _emit(mres.model_dump(), compact=args.compact, out_path=args.out)
        return 0 if mres.ok else 1
    if args.cmd == "crawl":
        cres = await safe_crawl_site(
            url=args.url,
            query=str(getattr(args, "query", "") or ""),
            max_pages=int(args.max_pages),
            max_chars_per_page=int(args.max_chars),
            concurrency=int(args.concurrency),
            timeout_sec=float(args.timeout),
            disk_cache=server.disk_cache,
            use_cache=not getattr(args, "no_cache", False),
        )
        _emit(cres.model_dump(), compact=args.compact, out_path=args.out)
        return 0 if cres.ok else 1
    if args.cmd == "batch":
        domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else ["auto"]
        tasks = [
            server.pool.search(query=q, domains=domains, limit=args.limit, budget_ms=args.budget_ms)
            for q in args.queries[:16]
        ]
        outs = await asyncio.gather(*tasks)
        payload = {"version": __version__, "query_count": len(outs), "items": [o.model_dump() for o in outs]}
        _emit(payload, compact=args.compact, out_path=args.out)
        return 0
    if args.cmd == "research":
        if args.merge:
            report = await merge_lane_results(
                work_dir=args.merge,
                verify_top_k=args.verify_top_k,
                budget_ms=args.budget_ms,
                render_markdown=bool(args.md),
                render_bibtex=bool(args.bib),
            )
            if args.md:
                md_path = pathlib.Path(args.md)
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(report.markdown or render_research_markdown(report), encoding="utf-8")
                print(f"[omni] markdown brief -> {md_path}", file=sys.stderr)
            if args.bib:
                bib_path = pathlib.Path(args.bib)
                bib_path.parent.mkdir(parents=True, exist_ok=True)
                bib_path.write_text(report.bibtex or render_citations_bibtex(report), encoding="utf-8")
                print(f"[omni] bibtex citations -> {bib_path}", file=sys.stderr)
            excl = set()
            if args.md:
                excl.add("markdown")
            if args.bib:
                excl.add("bibtex")
            _emit(report.model_dump(exclude=excl or None), compact=args.compact, out_path=args.out)
            return 0 if report.candidates else 4

        if not args.brief:
            print("[omni] ERROR: research requires <brief> unless --merge <dir> is provided", file=sys.stderr)
            return 2

        prev_report: ResearchReport | None = None
        if getattr(args, "diff", None):
            prev_path = pathlib.Path(args.diff)
            prev_report = ResearchReport.model_validate(json.loads(prev_path.read_text(encoding="utf-8")))
            # 纯离线双快照对比模式：brief 为已存在的 .json 报告路径
            maybe_curr = pathlib.Path(args.brief)
            if maybe_curr.suffix.lower() == ".json" and maybe_curr.exists():
                report = ResearchReport.model_validate(json.loads(maybe_curr.read_text(encoding="utf-8")))
                report.diff = diff_research_reports(prev_report, report)
                if args.md:
                    md_path = pathlib.Path(args.md)
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    md_path.write_text(render_research_markdown(report), encoding="utf-8")
                    print(f"[omni] markdown brief -> {md_path}", file=sys.stderr)
                if args.bib:
                    bib_path = pathlib.Path(args.bib)
                    bib_path.parent.mkdir(parents=True, exist_ok=True)
                    bib_path.write_text(render_citations_bibtex(report), encoding="utf-8")
                    print(f"[omni] bibtex citations -> {bib_path}", file=sys.stderr)
                excl = set()
                if args.md:
                    excl.add("markdown")
                if args.bib:
                    excl.add("bibtex")
                _emit(report.model_dump(exclude=excl or None), compact=args.compact, out_path=args.out)
                return 0

        domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
        terms = [t.strip() for t in args.terms.split(",") if t.strip()] if args.terms else None
        if args.export_lanes:
            plan = export_research_workorders(
                out_dir=args.export_lanes,
                brief=args.brief,
                domains=domains,
                mode=args.mode,
                extra_terms=terms,
                include_local=bool(args.include_local),
            )
            _emit(plan.model_dump(), compact=args.compact, out_path=args.out)
            return 0
        if args.plan_only:
            plan = plan_research_lanes(
                args.brief,
                domains=domains,
                mode=args.mode,
                extra_terms=terms,
                include_local=bool(args.include_local),
            )
            _emit(plan.model_dump(), compact=args.compact, out_path=args.out)
            return 0
        report = await run_research_rounds(
            pool=server.pool,
            brief=args.brief,
            domains=domains,
            mode=args.mode,
            extra_terms=terms,
            limit=args.limit,
            verify_top_k=args.verify_top_k,
            budget_ms=args.budget_ms,
            render_markdown=False,
            render_bibtex=bool(args.bib),
            include_local=bool(args.include_local),
            deepen=bool(args.deepen),
        )
        if prev_report is not None:
            report.diff = diff_research_reports(prev_report, report)
        if args.md:
            md_path = pathlib.Path(args.md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_text = render_research_markdown(report)
            md_path.write_text(md_text, encoding="utf-8")
            print(f"[omni] markdown brief -> {md_path}", file=sys.stderr)
        if args.bib:
            bib_path = pathlib.Path(args.bib)
            bib_path.parent.mkdir(parents=True, exist_ok=True)
            bib_path.write_text(report.bibtex or render_citations_bibtex(report), encoding="utf-8")
            print(f"[omni] bibtex citations -> {bib_path}", file=sys.stderr)
        excl = set()
        if args.md:
            excl.add("markdown")
        if args.bib:
            excl.add("bibtex")
        out_payload = (
            compact_research_report(report)
            if getattr(args, "concise", False)
            else report.model_dump(exclude=excl or None)
        )
        _emit(out_payload, compact=args.compact, out_path=args.out)
        return 0 if report.candidates else 4
    if args.cmd == "index":
        db_target = pathlib.Path(args.db) if args.db else server.index_db_path
        stats = await asyncio.to_thread(
            index_paths,
            args.paths,
            db_target,
            args.reset,
            server.embedder,
            getattr(args, "collection", "default") or "default",
        )
        _emit(stats.model_dump(), compact=args.compact, out_path=args.out)
        return 0
    if args.cmd == "watch":
        db_target = pathlib.Path(args.db) if args.db else server.index_db_path
        return await _run_watch(
            paths=args.paths,
            db_target=db_target,
            interval=float(args.interval),
            max_rounds=int(args.max_rounds),
            force_poll=bool(getattr(args, "poll", False)),
            embedder=server.embedder,
            collection=getattr(args, "collection", "default") or "default",
        )
    if args.cmd == "monitor":
        doms = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
        state_dir = pathlib.Path(args.state_dir) if args.state_dir else (PROJECT_ROOT / ".omni-monitors")
        mres = await run_monitor_check(
            pool=server.pool,
            topic=args.topic,
            state_dir=state_dir,
            mode=args.mode,
            domains=doms,
            limit=args.limit,
            verify_top_k=args.verify_top_k,
            budget_ms=args.budget_ms,
        )
        if args.md:
            md_path = pathlib.Path(args.md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(render_monitor_markdown(mres), encoding="utf-8")
            print(f"[omni] monitor markdown -> {md_path}", file=sys.stderr)
        _emit(mres.model_dump(), compact=args.compact, out_path=args.out)
        return 0
    if args.cmd == "verify":
        claim_objs: list[ClaimInput] = []
        md_text = ""
        if getattr(args, "file", None):
            fpath = pathlib.Path(args.file)
            md_text = fpath.read_text(encoding="utf-8", errors="replace")
        if getattr(args, "claim", None) and getattr(args, "url", None):
            claim_objs.append(ClaimInput(claim=str(args.claim), url=str(args.url)))
        if not claim_objs and not md_text.strip():
            print("[omni] verify 需要指定 Markdown 文件路径，或同时提供 --claim 与 --url", file=sys.stderr)
            return 2
        vrep = await verify_claims(
            claims=claim_objs,
            markdown=md_text or None,
            concurrency=int(args.concurrency),
            timeout_sec=float(args.timeout),
            max_chars=int(args.max_chars),
            disk_cache=server.disk_cache,
            use_cache=not getattr(args, "no_cache", False),
            render_markdown=bool(args.md),
        )
        if args.md:
            md_path = pathlib.Path(args.md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(vrep.markdown or render_verification_markdown(vrep), encoding="utf-8")
            print(f"[omni] verify markdown -> {md_path}", file=sys.stderr)
        _emit(vrep.model_dump(exclude={"markdown"}), compact=args.compact, out_path=args.out)
        return 0 if (vrep.unsupported_count == 0 and vrep.unreachable_count == 0 and vrep.blocked_count == 0) else 1
    return 2


async def _run_watch(
    paths: list[str],
    db_target: pathlib.Path,
    interval: float,
    max_rounds: int,
    force_poll: bool,
    embedder: Any,
    collection: str = "default",
) -> int:
    """监听目录变更并自动增量更新本地索引（优先使用可选 watchdog OS 事件，默认回退定时轮询）。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    use_watchdog = False
    ObserverCls = None
    HandlerBase = None
    if not force_poll and max_rounds == 0:
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore

            ObserverCls = Observer
            HandlerBase = FileSystemEventHandler
            use_watchdog = True
        except ImportError:
            use_watchdog = False

    if use_watchdog and ObserverCls is not None and HandlerBase is not None:
        print("[omni] watch mode=watchdog (OS filesystem events)", file=sys.stderr)
        stats = await asyncio.to_thread(index_paths, paths, db_target, False, embedder, collection)
        print(json.dumps(stats.model_dump(), ensure_ascii=False), flush=True)

        loop = asyncio.get_running_loop()
        dirty = asyncio.Event()
        db_name = db_target.name
        skip_parts = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".workbuddy"}

        class _DebounceHandler(HandlerBase):  # type: ignore[misc,valid-type]
            def on_any_event(self, event: Any) -> None:
                src = str(getattr(event, "src_path", "") or "")
                if not src:
                    return
                p = pathlib.Path(src)
                if p.name.startswith((db_name, ".omni-cache.db")):
                    return
                if any(part in skip_parts for part in p.parts):
                    return
                loop.call_soon_threadsafe(dirty.set)

        observer = ObserverCls()
        handler = _DebounceHandler()
        scheduled = 0
        for raw_p in paths:
            rp = pathlib.Path(raw_p).resolve()
            watch_dir = rp if rp.is_dir() else rp.parent
            if watch_dir.exists():
                observer.schedule(handler, str(watch_dir), recursive=True)
                scheduled += 1
        if scheduled > 0:
            observer.start()
            debounce_s = min(0.8, max(0.2, interval / 5.0))
            try:
                while True:
                    await dirty.wait()
                    await asyncio.sleep(debounce_s)
                    dirty.clear()
                    st = await asyncio.to_thread(index_paths, paths, db_target, False, embedder, collection)
                    if st.indexed > 0 or st.removed > 0:
                        print(json.dumps(st.model_dump(), ensure_ascii=False), flush=True)
            finally:
                observer.stop()
                observer.join(timeout=2.0)
            return 0

    print(f"[omni] watch mode=poll interval={max(0.5, interval)}s", file=sys.stderr)
    rounds = 0
    while True:
        stats = await asyncio.to_thread(index_paths, paths, db_target, False, embedder, collection)
        if rounds == 0 or stats.indexed > 0 or stats.removed > 0:
            print(json.dumps(stats.model_dump(), ensure_ascii=False), flush=True)
        rounds += 1
        if max_rounds and rounds >= max_rounds:
            break
        await asyncio.sleep(max(0.5, interval))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="omni", description="Search All in One — 面向 AI Agent 的万物元搜索引擎")
    parser.add_argument("--version", action="version", version=f"search-all-in-one {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="执行跨领域万物元搜索")
    p_search.add_argument("query", help="检索关键词")
    p_search.add_argument("-d", "--domains", default="auto", help="域列表（逗号分隔）：auto/all/web/news/knowledge/academic/code/code_evidence/packages/security/media/local/local_code")
    p_search.add_argument("-l", "--limit", type=int, default=6, help="每源条数上限")
    p_search.add_argument("--max-results", type=int, default=25, help="融合后最大条数")
    p_search.add_argument("--budget-ms", type=int, default=25000, help="总预算毫秒")
    p_search.add_argument("--no-cache", action="store_true", help="绕过内存与磁盘 TTL 缓存强制实时查询")
    p_search.add_argument("--no-relax", action="store_true", help="禁用长查询 0 命中自适应宽词补查（严格精确查询）")
    p_search.add_argument("--concise", action="store_true", help="省 Token 精简输出视图")
    p_search.add_argument("--compact", action="store_true", help="单行紧凑 JSON")
    p_search.add_argument("--out", help="输出到 UTF-8 JSON 文件")

    p_fetch = sub.add_parser("fetch", help="安全抓取并抽取公网 URL 正文（支持单条或多条并发批量抓取）")
    p_fetch.add_argument("urls", nargs="+", help="一个或多个目标 URL")
    p_fetch.add_argument("-q", "--query", default="", help="可选聚焦查询词（长文档按段级相关性裁剪保留核心内容）")
    p_fetch.add_argument("--max-chars", type=int, default=8000, help="正文截断上限")
    p_fetch.add_argument("--timeout", type=float, default=15.0, help="超时秒数")
    p_fetch.add_argument("--concurrency", type=int, default=5, help="批量模式最大并发数（1-10）")
    p_fetch.add_argument("--llms-txt", action="store_true", help="对站点根路径优先探测 /llms.txt")
    p_fetch.add_argument("--no-cache", action="store_true", help="绕过磁盘 TTL 缓存强制实时抓取")
    p_fetch.add_argument("--compact", action="store_true")
    p_fetch.add_argument("--out")

    p_map = sub.add_parser("map", help="受控站点目录映射（/llms.txt -> sitemap.xml -> 同源 HTML 链接）")
    p_map.add_argument("url", help="目标站点 URL")
    p_map.add_argument("-q", "--query", default="", help="可选过滤/排序查询词")
    p_map.add_argument("--max-urls", type=int, default=50, help="最大返回 URL 数（默认 50）")
    p_map.add_argument("--timeout", type=float, default=12.0, help="超时秒数")
    p_map.add_argument("--compact", action="store_true")
    p_map.add_argument("--out")

    p_batch = sub.add_parser("batch", help="并发批量查询")
    p_batch.add_argument("queries", nargs="+", help="多条查询词")
    p_batch.add_argument("-d", "--domains", default="auto")
    p_batch.add_argument("-l", "--limit", type=int, default=5)
    p_batch.add_argument("--budget-ms", type=int, default=30000)
    p_batch.add_argument("--compact", action="store_true")
    p_batch.add_argument("--out")

    p_res = sub.add_parser("research", help="多模式全网深度调研编排（差异化发现 + 缺口补查 + 原文核验 + 多 Agent 合单 + Markdown 简报）")
    p_res.add_argument("brief", nargs="?", default=None, help="研究主题简述（使用 --merge 时可省略）")
    p_res.add_argument("-m", "--mode", choices=["standard", "feasibility", "benchmark", "factcheck"], default="standard", help="调研模式")
    p_res.add_argument("--terms", default=None, help="附加核心词（逗号分隔）")
    p_res.add_argument("-d", "--domains", default=None, help="限定域列表（逗号分隔，默认全视角）")
    p_res.add_argument("-l", "--limit", type=int, default=4, help="每源条数上限")
    p_res.add_argument("--verify-top-k", type=int, default=3, help="自动抓取核验前 K 条不同站点原文")
    p_res.add_argument("--budget-ms", type=int, default=35000, help="总预算毫秒")
    p_res.add_argument("--plan-only", action="store_true", help="仅打印带 plan_id 的 lane 计划（不联网）")
    p_res.add_argument("--export-lanes", default=None, help="导出多 Agent 并行工单目录（plan.json + agents/*.md，不联网）")
    p_res.add_argument("--merge", default=None, help="按 plan.json 严格合并目录下多 Agent 的 *.result.json")
    p_res.add_argument("--diff", default=None, help="与历史调研报告 JSON 快照做变更对比（支持联网增量对比或离线双快照对比）")
    p_res.add_argument("--md", default=None, help="输出结构化中文 Markdown 调研简报路径")
    p_res.add_argument("--bib", default=None, help="导出规范 BibTeX 引用文献文件路径")
    p_res.add_argument("--include-local", action="store_true", help="将本机知识库 (local_kb) 纳入联合调研并做内外资产对照")
    p_res.add_argument("--deepen", action="store_true", help="开启确定性自反追问（按首轮共振新术语自动深挖 1 轮）")
    p_res.add_argument("--concise", action="store_true", help="省 Token 精简输出视图")
    p_res.add_argument("--compact", action="store_true")
    p_res.add_argument("--out")

    p_idx = sub.add_parser("index", help="构建或增量更新本地多模态文件索引")
    p_idx.add_argument("paths", nargs="+", help="要索引的目录或文件路径")
    p_idx.add_argument("--db", default=None, help="自定义索引数据库路径（默认 .omni-index.db）")
    p_idx.add_argument("--collection", default="default", help="所属集合命名空间（如 work/notes，默认 default）")
    p_idx.add_argument("--reset", action="store_true", help="清空重建索引")
    p_idx.add_argument("--compact", action="store_true")
    p_idx.add_argument("--out")

    p_watch = sub.add_parser("watch", help="持续监听目录变更并自动增量更新本地索引（可选 watchdog OS 事件 / 默认轮询）")
    p_watch.add_argument("paths", nargs="+", help="要监听的目录路径")
    p_watch.add_argument("--db", default=None, help="自定义索引数据库路径")
    p_watch.add_argument("--collection", default="default", help="所属集合命名空间（默认 default）")
    p_watch.add_argument("--interval", type=float, default=5.0, help="轮询间隔秒数（默认 5s）")
    p_watch.add_argument("--max-rounds", type=int, default=0, help="最大轮次（0=持续运行，便于脚本受控退出）")
    p_watch.add_argument("--poll", action="store_true", help="强制使用轮询模式（跳过 watchdog OS 事件监听）")

    p_doc = sub.add_parser("doctor", help="环境与引擎健康体检")
    p_doc.add_argument("--probe", action="store_true", help="向 L0 源发起轻量实时探活")
    p_doc.add_argument("--reset-health", action="store_true", help="体检前清空全部历史熔断冷却状态")
    p_doc.add_argument("--reset-engines", default="", help="精准重置指定引擎 ID 的熔断冷却状态（逗号分隔，如 codeberg,gitee）")
    p_doc.add_argument("--stats", action="store_true", help="输出近 N 天各引擎成功率、P50/P95 延迟、Top-5 贡献与调优建议")
    p_doc.add_argument("--stats-days", type=int, default=7, help="遥测统计窗口天数（默认 7 天）")
    p_doc.add_argument("--compact", action="store_true")
    p_doc.add_argument("--out")

    p_mon = sub.add_parser("monitor", help="主题持续增量监控（自动维护快照并仅输出新增 T1/变更摘要）")
    p_mon.add_argument("topic", help="持续监控的主题或组件名称")
    p_mon.add_argument("-m", "--mode", choices=["standard", "feasibility", "benchmark", "factcheck"], default="standard")
    p_mon.add_argument("-d", "--domains", default=None, help="限定域列表（逗号分隔）")
    p_mon.add_argument("-l", "--limit", type=int, default=4)
    p_mon.add_argument("--verify-top-k", type=int, default=2)
    p_mon.add_argument("--budget-ms", type=int, default=25000)
    p_mon.add_argument("--state-dir", default=None, help="历史快照目录（默认 .omni-monitors）")
    p_mon.add_argument("--md", default=None, help="可选输出增量变更 Markdown 路径")
    p_mon.add_argument("--compact", action="store_true")
    p_mon.add_argument("--out")

    p_crawl = sub.add_parser("crawl", help="站内定向采集（map 发现排序 -> 并发安全抓取 Top-N 正文）")
    p_crawl.add_argument("url", help="目标文档站点起始 URL")
    p_crawl.add_argument("-q", "--query", default="", help="可选聚焦查询词（用于排序站内页面并裁剪长文）")
    p_crawl.add_argument("--max-pages", type=int, default=5, help="最大采集页数（1-15，默认 5）")
    p_crawl.add_argument("--max-chars", type=int, default=4000, help="每页最大抽取字符数（默认 4000）")
    p_crawl.add_argument("--concurrency", type=int, default=4, help="并发抓取上限（1-10）")
    p_crawl.add_argument("--timeout", type=float, default=15.0, help="单页超时秒数")
    p_crawl.add_argument("--no-cache", action="store_true", help="绕过磁盘 TTL 缓存强制实时抓取")
    p_crawl.add_argument("--compact", action="store_true")
    p_crawl.add_argument("--out")

    p_ver = sub.add_parser("verify", help="主张级引用核验（校验 Markdown 报告或单条 claim+url 是否被原文支撑，消除引用幻觉）")
    p_ver.add_argument("file", nargs="?", default=None, help="待核验的 Markdown 文件路径")
    p_ver.add_argument("--claim", default=None, help="单条待核验主张文本（需配合 --url）")
    p_ver.add_argument("--url", default=None, help="单条主张对应的引用 URL")
    p_ver.add_argument("--concurrency", type=int, default=5, help="并发抓取上限（1-10）")
    p_ver.add_argument("--timeout", type=float, default=15.0, help="单条抓取超时秒数")
    p_ver.add_argument("--max-chars", type=int, default=12000, help="每页正文最大抽取字符数")
    p_ver.add_argument("--no-cache", action="store_true", help="绕过磁盘 TTL 缓存强制实时抓取")
    p_ver.add_argument("--md", default=None, help="输出 Markdown 核验审计表路径")
    p_ver.add_argument("--compact", action="store_true")
    p_ver.add_argument("--out")

    p_mcp = sub.add_parser("mcp", help="启动 MCP Server（默认 stdio；可选 --transport http 本机回环常驻共享）")
    p_mcp.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="传输模式：stdio（默认）或 http（Streamable HTTP）")
    p_mcp.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址（默认 127.0.0.1，非回环必须配合 --token）")
    p_mcp.add_argument("--port", type=int, default=8765, help="HTTP 监听端口（默认 8765）")
    p_mcp.add_argument("--token", default="", help="可选 Bearer 鉴权令牌（也可通过 OMNI_MCP_TOKEN 环境变量配置）")

    args = parser.parse_args(argv)
    rc = asyncio.run(_run(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
