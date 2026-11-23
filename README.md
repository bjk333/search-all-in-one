# Search All in One（`omni-search`）

面向 **AI Agent** 的可审计、失败感知、声明式万物元搜索引擎（CLI + MCP + Skill，全局 CLI 支持 `search-all-in-one` 与短命令 `omni`）。

本仓库独立孵化于 `multi-code-platform-searcher` 的工程契约体系，完整继承其诚实覆盖率（`coverage.verdict` / `reason_codes` / `0 命中 ≠ 不存在`）、自适应熔断、预算控制、输出防注入清洗与 MCP `structuredContent`，并已实现 **100% 原生 Python `asyncio` 替代原仓全部代码托管、包生态、跨仓代码片段证据与免建库本机代码检索**（共 **50 个原生引擎、11 大能力域**，0 PowerShell 子进程；旧 `mcps_bridge` 保留为 `OMNI_ENABLE_MCPS_BRIDGE=1` 可选兼容开关）。

## 核心能力

- **11 大检索域 + 安全原文读取**：`web` / `news`（HackerNews/Google News RSS/GDELT）/ `knowledge`（中英维基/Wikidata/MDN/IETF RFC/StackExchange/OSM）/ `academic`（arXiv/HuggingFace Papers/OpenAlex/Crossref/Zenodo/Europe PMC）/ `code`（GitHub/GitHub Issues/GitLab/Codeberg/Gitee）/ **`code_evidence`**（grep.app 官方 MCP / Sourcegraph SSE / GitHub Code）/ `packages`（PyPI/Go/npm/crates/Maven/NuGet/RubyGems/Packagist/DockerHub/MCP Registry/Anaconda/Terraform Registry）/ **`security`**（NIST NVD CVE 2.0 + Google OSV.dev 双源）/ `media`（HuggingFace Models & Datasets/Wikimedia Commons/OpenLibrary/Archive.org）/ **`local`**（单文件 SQLite WAL + SHA-256 内容哈希增量 + 多 `collection:`/`tag:` 隔离 + FTS5 BM25F + 结构感知分块向量混合索引，支持 PDF 逐页定位与扫描件识别、EPUB 章节、CSV 表头分块、一跳引用图谱与纯 Python 媒体头元数据检索）/ **`local_code`**（免建库 `rg --json` / `ast-grep` 实时代码检索） + 安全正文抽取 `fetch`
- **三层合规源**：
  - `L0`（零配置免 Key 开放 API）：DuckDuckGo Instant、Wikipedia（中/英）、Wikidata、MDN Web Docs、StackExchange、OpenStreetMap、HackerNews、GDELT、OpenAlex、arXiv、Crossref、HuggingFace、OpenLibrary、Internet Archive、GitHub/GitLab/Codeberg/Gitee、grep.app、Sourcegraph、npm/crates.io/Maven/NuGet/RubyGems/Packagist/DockerHub/MCP Registry
  - `L1`（自托管隐私全网聚合）：SearXNG（仓库内置 [docker-compose.searxng.yml](docker-compose.searxng.yml) 与开启 JSON 输出的 `deploy/searxng/settings.yml`，启动后配置 `SEARXNG_URL=http://127.0.0.1:8080`）
  - `L2`（BYOK 商业/认证搜索）：DeepSeek Search（自动只读复用原仓 `.env` 中的 `DEEPSEEK_API_KEY`，支持 402 `quota-exhausted` 与 `rerank_score >= 1.0` 质量过滤）/ Brave Search / Tavily / GitHub Code Search（按需配置 `GITHUB_TOKEN`）
- **诚实覆盖率契约与精准自愈**：引擎不自报状态，由池侧统一派生 `full / partial / degraded / none / no-engines`，0 命中且存在阻塞/降级时显式标出 `low_confidence_negative=true`；支持半开熔断探活、`omni doctor` 健康汇总与可执行修复建议，以及 `omni doctor --reset-engines <ids>` 按引擎精准重置熔断；长查询 0 命中自动宽词补查（可 `--no-relax` / `autoRelax=false` 关闭）
- **安全正文抽取 (`omni fetch`)**：内置逐跳前置 SSRF/内网 IP/元数据地址拦截、HTML → 结构化 Markdown 抽取（优先 `<article>`/`<main>`、代码块语言围栏、外链保真、剥离 `<svg>`/`<header>`/`<nav>`）、`Accept: text/markdown` 优先协商（含 frontmatter/`# H1` 标题抽取、`Content-Signal` 与 `x-markdown-tokens` Agentic 响应头解析）、**9 大高频平台官方公开 API 直取**（GitHub README/Issues/PR/Blob、npm、PyPI、arXiv、crates.io、HuggingFace Models/Datasets、Wikipedia REST v1、DOI Crossref/arXiv、Hacker News Firebase）、可选 `/llms.txt` → `/llms-full.txt` 探测（`--llms-txt`）、磁盘 TTL 缓存（可 `--no-cache`）、并发批量抓取、公网 PDF 文献直抽（`pypdf`）、体积上限与 `untrusted_content=true` 数据边界标记
- **8 个正交 MCP 工具与 4 核心依赖**：`omni_search` / `omni_fetch` / `omni_batch` / `omni_research` / `omni_index` / `omni_doctor` / `omni_monitor` / `omni_verify`，核心运行时仅依赖 `httpx` / `pyyaml` / `pydantic` / `pypdf`

## 快速开始

```bash
uv sync --extra dev
uv run omni doctor --stats
uv run omni doctor --reset-engines codeberg,gitee
uv run python evals/ir_benchmark.py
uv run omni search "multimodal search" -d knowledge,news,academic --limit 3 --concise
uv run omni search "exact technical phrase" -d code --no-relax --compact
uv run omni index "D:/01 项目/omni-search"
uv run omni watch "D:/01 项目/omni-search" --interval 5   # 安装 uv sync --extra watch 后自动启用 OS 事件监听，--poll 可强制轮询
uv run omni search "type:code fuse_results_rrf" -d local --limit 5
uv run omni search "type:image 1920x1080" -d local --limit 5
uv run omni search "声明式引擎" -d local,knowledge --limit 5
uv run omni research "universal multimodal search" --mode feasibility --terms "metasearch,mcp" --include-local --deepen --verify-top-k 3 --md report.md --bib refs.bib
uv run omni research "universal multimodal search" --diff prev.json --md report_diff.md
uv run omni research --diff prev.json curr.json --md offline_diff.md
uv run omni research "universal multimodal search" --mode factcheck --export-lanes ./rr-work
uv run omni research --merge ./rr-work --verify-top-k 3 --md ./rr-work/brief.md --bib ./rr-work/refs.bib
uv run omni monitor "sqlite-vec" --domains code,academic --md ./monitor.md
uv run omni map "https://docs.crawl4ai.com" --query "markdown" --max-urls 20
uv run omni crawl "https://docs.crawl4ai.com" --query "markdown" --max-pages 4
uv run omni fetch "https://arxiv.org/pdf/2608.24060v1" --query "FTS5 hybrid" --max-chars 2000
uv run omni fetch "https://www.sqlite.org/fts5.html" "https://modelcontextprotocol.io" --concurrency 4
uv run omni verify ./report.md --md ./verify_audit.md
```

## 接入 MCP 客户端

支持标准 `stdio`（默认）以及多客户端共享连接池/缓存的 **MCP Streamable HTTP**（本机回环常驻：`omni mcp --transport http --host 127.0.0.1 --port 8765`，支持 `--token` / `OMNI_MCP_TOKEN` Bearer 鉴权与 SSE 进度流）：

```json
{
  "mcpServers": {
    "search-all-in-one": {
      "command": "uv",
      "args": ["--directory", "D:/01 项目/omni-search", "run", "omni", "mcp"]
    }
  }
}
```

## 零代码扩展新数据源

只需在 `engines/<domain>/<id>.yaml` 新增一份声明式配置（无需修改调度与融合代码），`omni doctor` 与测试门会自动加载校验。

## 本地混合检索配置（可选真离线静态向量 / 外部稠密向量）

默认内置零依赖确定性子词向量（`hash_ngram_v1_256`）+ **结构感知分块索引（PDF 逐页 `P<n>` / PPTX 逐页 `S<n>` / 代码与 Markdown 标题函数边界 `L<start>-L<end>`）**，与 FTS5 BM25 做双路 RRF 融合；如需升级到真语义向量，可在 `.env` 二选一配置（切换模型后首次 `omni index` 自动增量补算全部 chunk 向量）：

1. **纯 CPU 离线静态语义向量（Model2Vec，无需 GPU / Ollama）**：
   先安装可选依赖 `uv sync --extra semantic`，再配置：
   ```env
   OMNI_EMBED_BACKEND=model2vec
   OMNI_EMBED_MODEL=minishlab/potion-multilingual-128M
   ```
2. **本机 Ollama 或任意 OpenAI 兼容 `/v1/embeddings` 服务**：
   ```env
   OMNI_EMBED_URL=http://127.0.0.1:11434/v1/embeddings
   OMNI_EMBED_MODEL=nomic-embed-text
   # OMNI_EMBED_API_KEY=sk-...
   # OMNI_EMBED_MIN_SCORE=0.35
   ```

## 可选跨源精排配置（`/v1/rerank`）

在 RRF 去重融合之后，如需接入标准 `/v1/rerank` 精排服务（兼容 Cohere / Jina / SiliconFlow / 本地 vLLM / BGE-Reranker），配置以下变量即可自动生效（未配置零开销；服务不可达时自动优雅回退原 RRF 序并记告警）：

```env
OMNI_RERANK_URL=http://127.0.0.1:8000/v1/rerank
OMNI_RERANK_MODEL=BAAI/bge-reranker-v2-m3
# OMNI_RERANK_API_KEY=sk-...
```

## 许可证

Apache-2.0
