import pytest

from omni_search.contracts import ReasonCode
from omni_search.local_index import LocalIndexEngine, index_paths
from omni_search.router import route_domains


def test_route_domains_detects_local_intent():
    assert "local" in route_domains("搜一下本地笔记和截图", ["auto"])
    assert "local" in route_domains("find my local files and pdf notes", ["auto"])
    assert route_domains("anything", ["local"]) == ["local"]


@pytest.mark.asyncio
async def test_local_engine_uninitialized_returns_not_run(tmp_path):
    eng = LocalIndexEngine(db_path=tmp_path / "missing.db")
    env = await eng.search("anything", limit=5, client=None)
    assert env.failure is not None
    assert env.failure.reason_code == ReasonCode.NOT_RUN
    assert env.results == []


@pytest.mark.asyncio
async def test_local_index_incremental_chinese_code_and_multimodal_search(tmp_path):
    ws = tmp_path / "docs"
    ws.mkdir()
    md_file = ws / "架构笔记.md"
    md_file.write_text("# 搜索万物引擎架构\n\n采用声明式 YAML 与本地 FTS5 混合检索。", encoding="utf-8")
    py_file = ws / "engine.py"
    py_file.write_text("class FederatedPool:\n    def fuse_rrf(self): pass\n", encoding="utf-8")
    img_file = ws / "architecture_diagram.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    vid_file = ws / "demo_walkthrough.mp4"
    vid_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    db_path = tmp_path / "omni.db"
    stats1 = index_paths([ws], db_path=db_path)
    assert stats1.indexed == 4
    assert stats1.skipped_unchanged == 0
    assert stats1.metadata_only == 2  # png + mp4
    assert stats1.total_docs == 4

    # 二次增量扫描：全部未改动应跳过
    stats2 = index_paths([ws], db_path=db_path)
    assert stats2.indexed == 0
    assert stats2.skipped_unchanged == 4

    eng = LocalIndexEngine(db_path=db_path)

    # 1. 中文全文检索
    env_zh = await eng.search("搜索引擎架构", limit=5, client=None)
    assert env_zh.failure is None
    assert any("架构笔记.md" in r.title for r in env_zh.results)
    assert env_zh.results[0].modality == "document"
    assert env_zh.results[0].url.startswith("file:///")

    # 2. 英文代码标识符检索
    env_code = await eng.search("FederatedPool", limit=5, client=None)
    assert env_code.failure is None
    assert any(r.title == "engine.py" and r.modality == "code" for r in env_code.results)

    # 3. 多模态图像元数据检索
    env_img = await eng.search("architecture_diagram", limit=5, client=None)
    assert env_img.failure is None
    assert any(r.title == "architecture_diagram.png" and r.modality == "image" for r in env_img.results)

    # 4. 删除文件后二次增量同步应自动清理
    img_file.unlink()
    stats3 = index_paths([ws], db_path=db_path)
    assert stats3.removed == 1
    assert stats3.total_docs == 3
    env_img2 = await eng.search("architecture_diagram", limit=5, client=None)
    assert env_img2.failure is None
    assert env_img2.results == []


@pytest.mark.asyncio
async def test_local_index_extracts_docx_pptx_xlsx_without_external_deps(tmp_path):
    import zipfile

    ws = tmp_path / "office"
    ws.mkdir()

    # 构造标准 .docx
    docx_p = ws / "方案评审.docx"
    with zipfile.ZipFile(docx_p, "w") as zf:
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>全网万物搜索引擎可行性评审结论</w:t></w:r></w:p></w:body></w:document>',
        )

    # 构造标准 .pptx
    pptx_p = ws / "roadmap.pptx"
    with zipfile.ZipFile(pptx_p, "w") as zf:
        zf.writestr(
            "ppt/slides/slide1.xml",
            '<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>OmniSearch Phase3 Multimodal Roadmap</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
        )

    db_path = ws / ".omni-index.db"
    st = index_paths([ws], db_path=db_path)
    assert st.indexed == 2
    assert st.metadata_only == 0
    # 索引库文件自身不得被自索引，二次扫描必须 0 增量
    st_again = index_paths([ws], db_path=db_path)
    assert st_again.indexed == 0
    assert st_again.total_docs == 2

    eng = LocalIndexEngine(db_path=db_path)
    r_doc = await eng.search("可行性评审", limit=5, client=None)
    assert any(x.title == "方案评审.docx" and x.modality == "document" and not x.extra["metadata_only"] for x in r_doc.results)

    r_ppt = await eng.search("Multimodal Roadmap", limit=5, client=None)
    assert any(x.title == "roadmap.pptx" and x.modality == "document" and not x.extra["metadata_only"] for x in r_ppt.results)


def _make_minimal_pdf(text: str) -> bytes:
    """构造标准最小单页 ASCII 文本 PDF 字节流（用于离线可复现 PDF 抽取测试）。"""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1"))
    return bytes(buf)


@pytest.mark.asyncio
async def test_local_index_extracts_real_pdf_text(tmp_path):
    ws = tmp_path / "pdfs"
    ws.mkdir()
    pdf_file = ws / "sqlite_hybrid_report.pdf"
    pdf_file.write_bytes(_make_minimal_pdf("SQLite FTS5 Hybrid Vector Search Architecture 2026"))

    db_path = tmp_path / "pdf.db"
    stats = index_paths([ws], db_path=db_path)
    assert stats.indexed == 1
    assert stats.metadata_only == 0

    eng = LocalIndexEngine(db_path=db_path)
    env = await eng.search("Hybrid Vector Search Architecture", limit=5, client=None)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].title == "sqlite_hybrid_report.pdf"
    assert env.results[0].extra["metadata_only"] is False


@pytest.mark.asyncio
async def test_local_index_hybrid_fts5_and_vector_rrf_recall(tmp_path):
    """内置向量路应补齐近似拼写/词形变化召回，并与 FTS5 双路 RRF 融合且披露 retrieval_mode。"""
    ws = tmp_path / "hybrid"
    ws.mkdir()
    (ws / "federation_design.md").write_text(
        "# Federated Metasearch Pool\n\nImplements reciprocal rank fusion and circuit breaker governance.",
        encoding="utf-8",
    )
    (ws / "cooking_recipe.md").write_text(
        "# Tomato Soup\n\nBoil fresh tomatoes with basil and olive oil for twenty minutes.",
        encoding="utf-8",
    )

    db_path = tmp_path / "hybrid.db"
    stats = index_paths([ws], db_path=db_path)
    assert stats.indexed == 2
    assert stats.vectors_updated == 2

    eng = LocalIndexEngine(db_path=db_path)

    # 1. 精确命中走双路 hybrid
    env_exact = await eng.search("Federated Metasearch reciprocal fusion", limit=5, client=None)
    assert env_exact.results
    top = env_exact.results[0]
    assert top.title == "federation_design.md"
    assert top.extra["retrieval_mode"] == "hybrid"
    assert top.extra["vector_score"] is not None and top.extra["vector_score"] > 0.25

    # 2. 拼写近似/词形变化（FTS5 精确词 0 命中）应由向量路兜底召回且拒绝无关食谱文档
    env_fuzzy = await eng.search("federating metasearching reciprocally", limit=5, client=None)
    assert any(r.title == "federation_design.md" and r.extra["retrieval_mode"] == "vector" for r in env_fuzzy.results)
    assert all(r.title != "cooking_recipe.md" for r in env_fuzzy.results)


@pytest.mark.asyncio
async def test_local_index_recomputes_vectors_when_embedder_changes(tmp_path):
    """切换可插拔 Embedder 时，即使文件未改动也应自动重算向量并按新语义空间检索。"""
    from omni_search.embeddings import Embedder

    class StubSemanticEmbedder:
        id = "stub-semantic-v1"
        dim = 4
        min_score = 0.5

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for t in texts:
                low = t.lower()
                if "汽车" in t or "vehicle" in low or "car" in low:
                    vecs.append([1.0, 0.0, 0.0, 0.0])
                elif "数据库" in t or "sqlite" in low:
                    vecs.append([0.0, 1.0, 0.0, 0.0])
                else:
                    vecs.append([0.0, 0.0, 1.0, 0.0])
            return vecs

    ws = tmp_path / "embed_switch"
    ws.mkdir()
    (ws / "auto_note.md").write_text("自动驾驶汽车控制方案", encoding="utf-8")
    (ws / "db_note.md").write_text("SQLite 单文件存储引擎", encoding="utf-8")
    db_path = tmp_path / "switch.db"

    # 先用默认 hash_ngram 建库
    st1 = index_paths([ws], db_path=db_path)
    assert st1.indexed == 2
    assert st1.embedder.startswith("hash_ngram")

    # 切换到 StubSemanticEmbedder：文件未变，但应自动重算 2 条向量
    stub = StubSemanticEmbedder()
    st2 = index_paths([ws], db_path=db_path, embedder=stub)
    assert st2.embedder == "stub-semantic-v1"
    assert st2.vectors_updated == 2

    # 跨语言纯语义查询："autonomous vehicle" 与中文 "自动驾驶汽车" 零字面重叠，依赖新向量召回
    eng = LocalIndexEngine(db_path=db_path, embedder=stub)
    env = await eng.search("autonomous vehicle", limit=5, client=None)
    assert len(env.results) == 1
    assert env.results[0].title == "auto_note.md"
    assert env.results[0].extra["retrieval_mode"] == "vector"
    assert env.results[0].extra["embedder"] == "stub-semantic-v1"


@pytest.mark.asyncio
async def test_local_index_degrades_gracefully_when_external_embedder_fails(tmp_path):
    """当外部向量服务不可达时，本地搜索必须优雅降级为纯 FTS5 并披露告警，不得整体报错。"""
    class FailingEmbedder:
        id = "openai_compat:bge-m3"
        dim = 4
        min_score = 0.35

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("connection refused")

    ws = tmp_path / "degrade"
    ws.mkdir()
    (ws / "note.md").write_text("SQLite FTS5 降级韧性测试文档", encoding="utf-8")
    db_path = tmp_path / "degrade.db"
    index_paths([ws], db_path=db_path)

    eng = LocalIndexEngine(db_path=db_path, embedder=FailingEmbedder())
    env = await eng.search("降级韧性", limit=5, client=None)
    assert env.failure is None
    assert len(env.results) == 1
    assert env.results[0].extra["retrieval_mode"] == "lexical"
    assert any("degraded to lexical FTS5" in w for w in env.warnings)


@pytest.mark.asyncio
async def test_local_index_query_centered_snippet_and_line_number(tmp_path):
    """关键词命中文件深处（第 95 行）时，snippet 应定位到命中窗口并输出 matched_line=95。"""
    ws = tmp_path / "deep"
    ws.mkdir()
    lines = [f"# filler header line {i} with padding content to push past 400 chars" for i in range(1, 95)]
    lines.append("def verify_circuit_breaker_threshold(failures: int) -> bool:")
    lines.append("    return failures >= 3")
    lines.extend([f"# trailing filler line {i}" for i in range(97, 115)])
    (ws / "breaker.py").write_text("\n".join(lines), encoding="utf-8")

    db_path = tmp_path / "deep.db"
    index_paths([ws], db_path=db_path)
    eng = LocalIndexEngine(db_path=db_path)

    env = await eng.search("verify_circuit_breaker_threshold", limit=5, client=None)
    assert env.failure is None
    assert len(env.results) == 1
    item = env.results[0]
    assert "verify_circuit_breaker_threshold" in item.snippet
    assert item.extra["matched_line"] == 95


@pytest.mark.asyncio
async def test_local_index_inline_modality_and_ext_filters(tmp_path):
    """支持内联 type:code / type:pdf / type:image / ext:.md 过滤且不污染查询词匹配。"""
    ws = tmp_path / "filters"
    ws.mkdir()
    (ws / "quantum_sim.py").write_text("def simulate_quantum_gate(): return 1\n", encoding="utf-8")
    (ws / "quantum_notes.md").write_text("# Quantum Notes\nNotes about quantum search.\n", encoding="utf-8")
    (ws / "quantum_paper.pdf").write_bytes(_make_minimal_pdf("Quantum Search Paper 2026"))
    (ws / "quantum_diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    db_path = tmp_path / "filters.db"
    index_paths([ws], db_path=db_path)
    eng = LocalIndexEngine(db_path=db_path)

    r_code = await eng.search("type:code quantum", limit=5, client=None)
    assert [x.title for x in r_code.results] == ["quantum_sim.py"]

    r_pdf = await eng.search("type:pdf quantum", limit=5, client=None)
    assert [x.title for x in r_pdf.results] == ["quantum_paper.pdf"]

    r_img = await eng.search("type:image quantum", limit=5, client=None)
    assert [x.title for x in r_img.results] == ["quantum_diagram.png"]

    r_md = await eng.search("ext:.md quantum", limit=5, client=None)
    assert [x.title for x in r_md.results] == ["quantum_notes.md"]


@pytest.mark.asyncio
async def test_openai_compat_embedder_real_http_end_to_end(tmp_path):
    """通过真实本地 HTTP Server 闭环验证 OpenAICompatEmbedder 协议、鉴权、重排与跨语言混合检索。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json
    import threading
    from omni_search.embeddings import OpenAICompatEmbedder

    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured["path"] = self.path
            captured["auth"] = self.headers.get("Authorization")
            captured["model"] = body.get("model")
            inputs = body.get("input") or []
            # 刻意倒序返回 index，验证客户端按 index 重排并 L2 归一化
            data = []
            for idx in reversed(range(len(inputs))):
                txt = str(inputs[idx]).lower()
                if "火箭" in txt or "rocket" in txt or "spacecraft" in txt:
                    vec = [3.0, 4.0, 0.0]  # L2 归一化后应为 [0.6, 0.8, 0.0]
                else:
                    vec = [0.0, 0.0, 5.0]  # L2 归一化后应为 [0.0, 0.0, 1.0]
                data.append({"index": idx, "embedding": vec})
            resp_bytes = json.dumps({"data": data}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        emb = OpenAICompatEmbedder(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="bge-m3",
            api_key="sk-local-test",
            min_score=0.5,
        )
        ws = tmp_path / "http_emb"
        ws.mkdir()
        (ws / "aerospace.md").write_text("可回收运载火箭推进系统设计", encoding="utf-8")
        (ws / "garden.md").write_text("阳台多肉植物浇水指南", encoding="utf-8")
        db_path = tmp_path / "http_emb.db"

        stats = index_paths([ws], db_path=db_path, embedder=emb)
        assert stats.vectors_updated == 2
        assert captured["path"] == "/v1/embeddings"
        assert captured["auth"] == "Bearer sk-local-test"
        assert captured["model"] == "bge-m3"

        eng = LocalIndexEngine(db_path=db_path, embedder=emb)
        env = await eng.search("reusable rocket spacecraft", limit=5, client=None)
        assert env.failure is None
        assert len(env.results) == 1
        assert env.results[0].title == "aerospace.md"
        assert env.results[0].extra["retrieval_mode"] == "vector"
        assert env.results[0].extra["vector_score"] == pytest.approx(1.0, abs=1e-3)
    finally:
        server.shutdown()
        server.server_close()


def _make_multipage_pdf(pages: list[str]) -> bytes:
    """构造合法多页 ASCII 文本 PDF（用于离线验证分块与精确页码定位）。"""
    objs: list[bytes] = []
    # obj 1: Catalog, obj 2: Pages, obj 3: Font
    page_obj_ids = [4 + i * 2 for i in range(len(pages))]
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {len(pages)} >> endobj\n".encode("latin-1"))
    objs.append(b"3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    for i, text in enumerate(pages):
        pid = 4 + i * 2
        cid = pid + 1
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
        objs.append(
            f"{pid} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >> endobj\n".encode("latin-1")
        )
        objs.append(
            f"{cid} 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n"
        )

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xp = len(buf)
    buf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xp}\n%%EOF\n".encode("latin-1"))
    return bytes(buf)


@pytest.mark.asyncio
async def test_local_index_multipage_pdf_chunk_locates_exact_page(tmp_path):
    """多页 PDF 按页分块向量化，命中第 3 页时应精确输出 matched_page=3、matched_locator='P3' 与页内容摘要。"""
    ws = tmp_path / "mpdf"
    ws.mkdir()
    pdf_bytes = _make_multipage_pdf(
        [
            "Page one covers indoor balcony gardening and watering schedules.",
            "Page two discusses Mediterranean olive oil tomato soup recipes.",
            "Page three introduces Distributed Consensus Raft Protocol 2026.",
        ]
    )
    (ws / "handbook.pdf").write_bytes(pdf_bytes)
    db_path = tmp_path / "mpdf.db"

    stats = index_paths([ws], db_path=db_path)
    assert stats.indexed == 1
    assert stats.chunks_indexed >= 3

    eng = LocalIndexEngine(db_path=db_path)
    env = await eng.search("Distributed Consensus Raft", limit=5, client=None)
    assert env.failure is None
    assert len(env.results) == 1
    item = env.results[0]
    assert item.extra["matched_locator"] == "P3"
    assert item.extra["matched_page"] == 3
    assert "[P3]" in item.snippet and "Raft" in item.snippet


@pytest.mark.asyncio
async def test_local_index_long_document_deep_chunk_vector_recall(tmp_path):
    """长文档深处（第 140 行之后）的概念通过分块向量不被前文稀释，能由近似词形向量路精准召回。"""
    ws = tmp_path / "longdoc"
    ws.mkdir()
    lines = [f"Section {i}: Routine inventory ledger entry for warehouse box alpha {i}." for i in range(1, 140)]
    lines.extend(
        [
            "## Resilience Architecture",
            "Adaptive circuit breaker with exponential backoff prevents cascading timeouts.",
            "Half-open probing restores healthy endpoints automatically.",
        ]
    )
    (ws / "manual.md").write_text("\n".join(lines), encoding="utf-8")
    db_path = tmp_path / "longdoc.db"

    stats = index_paths([ws], db_path=db_path)
    assert stats.chunks_indexed >= 4

    eng = LocalIndexEngine(db_path=db_path)
    # 全部为词形变化（FTS5 0 字面命中），全靠深处结构边界分块向量召回
    env = await eng.search("resilient circuits breakers exponentially backoffs", limit=5, client=None)
    assert len(env.results) == 1
    item = env.results[0]
    assert item.title == "manual.md"
    assert item.extra["retrieval_mode"] == "vector"
    assert item.extra["matched_locator"] == "L140-L142"
    assert "backoff" in item.snippet.lower() or "circuit" in item.snippet.lower()


@pytest.mark.asyncio
async def test_local_index_pptx_slide_locator(tmp_path):
    """多页 PPTX 应按幻灯片分块并输出精确页标 S2。"""
    import zipfile

    ws = tmp_path / "slides"
    ws.mkdir()
    pptx_p = ws / "deck.pptx"
    with zipfile.ZipFile(pptx_p, "w") as zf:
        zf.writestr(
            "ppt/slides/slide1.xml",
            '<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Quarterly Finance Overview</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
        )
        zf.writestr(
            "ppt/slides/slide2.xml",
            '<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Quantum Entanglement Telemetry Design</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
        )
    db_path = tmp_path / "slides.db"
    stats = index_paths([ws], db_path=db_path)
    assert stats.chunks_indexed == 2

    eng = LocalIndexEngine(db_path=db_path)
    env = await eng.search("Quantum Entanglement Telemetry", limit=5, client=None)
    assert len(env.results) == 1
    assert env.results[0].extra["matched_locator"] == "S2"
    assert "[S2]" in env.results[0].snippet


def test_model2vec_embedder_adapter_and_factory():
    """Model2VecEmbedder 适配器应支持离线 L2 归一化并通过环境变量工厂装配。"""
    from omni_search.embeddings import Model2VecEmbedder, build_default_embedder

    class _FakeStaticModel:
        def encode(self, texts: list[str]):
            return [[3.0, 4.0, 0.0] for _ in texts]

    emb = Model2VecEmbedder(model_name="minishlab/potion-multilingual-128M", model=_FakeStaticModel())
    vecs = emb.embed_texts(["hello", "world"])
    assert len(vecs) == 2
    assert vecs[0] == pytest.approx([0.6, 0.8, 0.0], abs=1e-5)
    assert emb.id == "model2vec:minishlab/potion-multilingual-128M"

    built = build_default_embedder({"OMNI_EMBED_BACKEND": "model2vec", "OMNI_EMBED_MODEL": "demo/potion"})
    assert isinstance(built, Model2VecEmbedder)
    assert built.id == "model2vec:demo/potion"
