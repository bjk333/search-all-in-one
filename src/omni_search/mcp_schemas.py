"""MCP Server 工具、资源与提示词元数据定义（遵循 MCP 2025-06-18+ outputSchema / title / annotations 规范）。"""

from __future__ import annotations

from typing import Any

_COMMON_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["version", "query", "domains", "results", "coverage", "completeness", "ledger"],
    "properties": {
        "version": {"type": "string"},
        "query": {"type": "string"},
        "domains": {"type": "array", "items": {"type": "string"}},
        "results": {"type": "array", "items": {"type": "object"}},
        "coverage": {"type": "object"},
        "completeness": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "ledger": {"type": "object"},
        "low_confidence_negative": {"type": "boolean"},
    },
    "additionalProperties": True,
}

_READ_ONLY_WEB_ANNOTATIONS: dict[str, bool] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}

_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "omni_search",
        "title": "Omni Multi-Domain Search",
        "description": "面向 AI Agent 的万物元搜索（50 原生引擎、11 域：web/news/knowledge/academic/code/code_evidence/packages/security/media/local/local_code），返回 RRF 融合结果与诚实覆盖率。",
        "annotations": dict(_READ_ONLY_WEB_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "domains": {"type": "array", "items": {"type": "string"}, "description": "目标域列表，默认 ['auto']，可选 all/web/news/knowledge/academic/code/code_evidence/packages/security/media/local/local_code"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 6},
                "maxResults": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
                "budgetMs": {"type": "integer", "minimum": 500, "maximum": 120000, "default": 25000},
                "concise": {"type": "boolean", "default": False, "description": "省 Token 精简视图（保留完整覆盖率与核心字段，压缩摘录与冗余 extra）"},
                "autoRelax": {"type": "boolean", "default": True, "description": "长查询导致多数公网源 0 命中时是否自动宽词补查（默认开启；严格精确检索可设 false）"},
            },
            "additionalProperties": False,
        },
        "outputSchema": _COMMON_OUTPUT_SCHEMA,
    },
    {
        "name": "omni_fetch",
        "title": "Safe Web Fetch, Map & Crawl",
        "description": "安全拉取单条或批量（urls，上限 20）公网 URL 正文（支持 query 聚焦裁剪与 PDF/GitHub/PyPI/npm 官方 API），或设置 mode='map' 映射站点文档目录树、mode='crawl' 一键站内定向采集 Top-N 正文，内置 SSRF 阻断与 untrusted_content 标记。",
        "annotations": dict(_READ_ONLY_WEB_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "单条目标公网 http/https URL（与 urls 二选一）"},
                "urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20, "description": "批量并发抓取 URL 列表（上限 20 条，单条失败隔离）"},
                "mode": {"type": "string", "enum": ["fetch", "map", "crawl"], "default": "fetch", "description": "fetch=读取正文，map=映射站点目录树，crawl=按 query 映射并并发采集站内 Top-N 正文"},
                "query": {"type": "string", "default": "", "description": "可选聚焦查询词（fetch/crawl 模式按相关性裁剪长文，map/crawl 模式按相关性排序 URL）"},
                "maxChars": {"type": "integer", "minimum": 200, "maximum": 50000, "default": 8000},
                "maxUrls": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50, "description": "仅 mode='map' 生效：最大返回 URL 数量"},
                "maxPages": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5, "description": "仅 mode='crawl' 生效：站内最大采集页数"},
                "concurrency": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5, "description": "批量抓取最大并发数"},
                "timeoutSec": {"type": "number", "minimum": 1, "maximum": 60, "default": 15},
                "preferLlmsTxt": {"type": "boolean", "default": False, "description": "对站点根路径优先探测 /llms.txt"},
                "noCache": {"type": "boolean", "default": False, "description": "绕过磁盘 TTL 缓存强制实时抓取"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["ok", "untrusted_content"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_batch",
        "title": "Batch Concurrent Search",
        "description": "并发执行多条万物检索查询（上限 16 条），逐条保留完整 coverage 与结果。",
        "annotations": dict(_READ_ONLY_WEB_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16},
                "domains": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "budgetMs": {"type": "integer", "minimum": 1000, "maximum": 120000, "default": 30000},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["version", "query_count", "items"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_research",
        "title": "Deep Multi-Lane Research",
        "description": "多模式全网深度调研编排：按模式（standard/feasibility/benchmark/factcheck）拆差异化 lane 并发发现 -> 0 命中宽词补查 -> 主机名多样性 Top-K 原文核验 -> T1/T2/T3 证据分级与可选 Markdown 简报。",
        "annotations": dict(_READ_ONLY_WEB_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "required": ["brief"],
            "properties": {
                "brief": {"type": "string", "description": "研究主题或问题简述"},
                "mode": {"type": "string", "enum": ["standard", "feasibility", "benchmark", "factcheck"], "default": "standard"},
                "terms": {"type": "array", "items": {"type": "string"}, "description": "可选附加核心关键词"},
                "domains": {"type": "array", "items": {"type": "string"}, "description": "限定域列表（可选）"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 4},
                "verifyTopK": {"type": "integer", "minimum": 0, "maximum": 8, "default": 3},
                "budgetMs": {"type": "integer", "minimum": 2000, "maximum": 120000, "default": 35000},
                "renderMarkdown": {"type": "boolean", "default": False, "description": "是否在返回中附带结构化中文 Markdown 调研简报"},
                "renderBibtex": {"type": "boolean", "default": False, "description": "是否在返回中附带规范 BibTeX 引用清单"},
                "includeLocal": {"type": "boolean", "default": False, "description": "是否纳入本机知识库 (local_kb) 做内外联合对照调研"},
                "deepen": {"type": "boolean", "default": False, "description": "是否开启确定性自反追问（按首轮多候选共振术语深挖 1 轮）"},
                "concise": {"type": "boolean", "default": False, "description": "省 Token 精简视图（压缩候选与核验冗余字段）"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["version", "plan_id", "brief", "mode", "lanes", "candidates", "verified_evidence", "tier_counts", "gaps"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_index",
        "title": "Local Knowledge Base Indexer",
        "description": "构建或增量更新本地多模态 SQLite FTS5 索引（文档/代码全文 + 图像/音视频/PDF 元数据），供 omni_search(domains=['local']) 毫秒检索。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "required": ["paths"],
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "待索引的本地目录或文件路径列表"},
                "reset": {"type": "boolean", "default": False, "description": "是否清空重建索引"},
                "collection": {"type": "string", "default": "default", "description": "集合命名空间（如 work/notes），供检索时 collection: 过滤"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["db_path", "indexed", "skipped_unchanged", "removed", "total_docs"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_doctor",
        "title": "Engine Health & Telemetry Doctor",
        "description": "体检：列出全部声明式与桥接引擎、凭据就绪状态、健康熔断冷却状态与禁用配置。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "probe": {"type": "boolean", "default": False, "description": "是否对 L0 源发起轻量探活"},
                "resetHealth": {"type": "boolean", "default": False, "description": "体检前清空全部历史熔断冷却状态"},
                "resetEngines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "精准重置指定引擎 ID 的熔断冷却状态（不影响其他引擎）",
                },
                "stats": {"type": "boolean", "default": False, "description": "是否附带近 N 天各引擎成功率/P95 延迟/Top-5 贡献与调优建议面板"},
                "statsDays": {"type": "integer", "minimum": 1, "maximum": 30, "default": 7, "description": "遥测统计窗口天数（默认 7）"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["version", "engine_count", "engines"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_monitor",
        "title": "Incremental Topic Monitor",
        "description": "主题持续增量监控：与本地历史调研快照做 diff，仅报告新增 T1 高权威证据、新增/移除条目与失效链接。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {"type": "string", "description": "追踪主题或技术关键词"},
                "mode": {"type": "string", "enum": ["standard", "feasibility", "benchmark", "factcheck"], "default": "standard"},
                "domains": {"type": "array", "items": {"type": "string"}, "description": "限定域列表（可选）"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 4},
                "verifyTopK": {"type": "integer", "minimum": 0, "maximum": 8, "default": 2},
                "budgetMs": {"type": "integer", "minimum": 2000, "maximum": 120000, "default": 25000},
                "stateDir": {"type": "string", "description": "可选快照存储目录（默认项目根 .omni-monitors）"},
                "renderMarkdown": {"type": "boolean", "default": False, "description": "是否附带 Markdown 变更简报"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["topic", "is_baseline", "has_changes", "total_candidates"],
            "additionalProperties": True,
        },
    },
    {
        "name": "omni_verify",
        "title": "Citation & Claim Verifier",
        "description": "主张级引用核验引擎：并发安全拉取 Markdown 报告或主张列表中的全部引用 URL，提取硬事实锚点（版本号/CVE/数字/术语）与原文对照，输出 supported/partial/unsupported/unreachable/blocked 审计判定，消除引用幻觉。",
        "annotations": dict(_READ_ONLY_WEB_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "markdown": {"type": "string", "description": "待核验的 Markdown 报告或草稿文本（自动抽取带公网 URL 的主张句）"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["claim", "url"],
                        "properties": {
                            "claim": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "description": "显式主张与引用 URL 对列表（可与 markdown 合并或单独传入）",
                },
                "concurrency": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "timeoutSec": {"type": "number", "minimum": 1, "maximum": 60, "default": 15},
                "maxChars": {"type": "integer", "minimum": 500, "maximum": 50000, "default": 12000},
                "noCache": {"type": "boolean", "default": False},
                "renderMarkdown": {"type": "boolean", "default": False, "description": "是否附带中文 Markdown 引用核验审计表"},
            },
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "required": ["version", "total_claims", "unique_urls", "untrusted_content", "items"],
            "additionalProperties": True,
        },
    },
]


_RESOURCES_SPEC: list[dict[str, Any]] = [
    {
        "uri": "omni://research/last-report",
        "name": "最近一次深度调研简报",
        "description": "本会话最近一次 omni_research 产出的中文 Markdown 证据简报",
        "mimeType": "text/markdown",
    },
    {
        "uri": "omni://doctor/telemetry",
        "name": "近 7 天引擎效费比遥测面板",
        "description": "各引擎成功率、P50/P95 延迟、Top-5 贡献与调优建议 JSON",
        "mimeType": "application/json",
    },
    {
        "uri": "omni://local/index/summary",
        "name": "本地知识库索引概览",
        "description": "本地 SQLite 混合索引的文档总数、集合分布与模态分布",
        "mimeType": "application/json",
    },
]

_PROMPTS_SPEC: list[dict[str, Any]] = [
    {
        "name": "feasibility_research",
        "description": "技术方案可行性深度调研模板（竞品、学术、风险与一手核验）",
        "arguments": [{"name": "topic", "description": "要调研的技术方案或产品方向", "required": True}],
    },
    {
        "name": "benchmark_comparison",
        "description": "多方案开源对标与评测基准模板",
        "arguments": [{"name": "topic", "description": "要对标的技术栈或工具类别", "required": True}],
    },
    {
        "name": "security_cve_audit",
        "description": "组件/依赖安全漏洞与 CVE 一手核验模板",
        "arguments": [{"name": "product", "description": "目标开源库、组件或 CVE 编号", "required": True}],
    },
]
