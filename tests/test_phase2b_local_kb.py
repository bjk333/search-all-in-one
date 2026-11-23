"""Phase 2B 红灯测试：SHA-256 内容哈希增量、collection/tag 隔离、扫描件 PDF/EPUB/CSV 表头分块、一跳链接图谱与老库迁移。"""

from __future__ import annotations

import os
import sqlite3
import zipfile

import pytest

from omni_search.local_index import LocalIndexEngine, index_paths


def _make_blank_pdf(path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)


def _make_minimal_epub(path) -> None:
    container_xml = """<?xml version="1.0"?>
    <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
    </container>"""
    opf_xml = """<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
      <manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>
      <spine><itemref idref="c1"/></spine>
    </package>"""
    ch1_xhtml = """<html xmlns="http://www.w3.org/1999/xhtml">
      <body><h1>Chapter One</h1><p>QuantumLattice epub chapter explains hybrid retrieval architecture.</p></body>
    </html>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", ch1_xhtml)


def test_sha256_content_hash_skips_reembed_on_mtime_touch(tmp_path):
    """mtime 变化但内容未变（如 git checkout / touch）时，SHA-256 命中应跳过重解析与向量重算。"""
    ws = tmp_path / "hash_ws"
    ws.mkdir()
    doc = ws / "notes.md"
    doc.write_text("# Search Notes\n\nDeterministic content for hash deduplication.\n", encoding="utf-8")
    db = tmp_path / "idx.db"

    s1 = index_paths([ws], db_path=db)
    assert s1.indexed == 1
    assert s1.vectors_updated == 1

    # 仅修改 mtime（模拟 git checkout / copy），内容完全不变
    st = doc.stat()
    os.utime(doc, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    s2 = index_paths([ws], db_path=db)
    assert s2.indexed == 0
    assert s2.vectors_updated == 0
    assert s2.skipped_unchanged == 1

    # 真正修改相同长度的内容时必须重索引
    doc.write_text("# Search Notes\n\nDeterministic content for hash REINDEXATION.\n", encoding="utf-8")
    s3 = index_paths([ws], db_path=db)
    assert s3.indexed == 1
    assert s3.vectors_updated == 1


@pytest.mark.asyncio
async def test_collections_and_frontmatter_tag_filters(tmp_path):
    """支持多 collection 命名空间隔离与 Markdown YAML Frontmatter tag: 过滤。"""
    work_dir = tmp_path / "work"
    notes_dir = tmp_path / "notes"
    work_dir.mkdir()
    notes_dir.mkdir()

    (work_dir / "rfc.md").write_text(
        "---\ntitle: Work Hybrid RFC\ntags: [rag, sqlite]\n---\n\nHybrid search architecture for work cluster.\n",
        encoding="utf-8",
    )
    (notes_dir / "journal.md").write_text(
        "---\ntitle: Personal Journal\ntags:\n  - architecture\n  - notes\n---\n\nHybrid search reflections in personal notes.\n",
        encoding="utf-8",
    )

    db = tmp_path / "multi.db"
    index_paths([work_dir], db_path=db, collection="work")
    index_paths([notes_dir], db_path=db, collection="notes")

    eng = LocalIndexEngine(db_path=db)
    r_work = await eng.search("collection:work hybrid", limit=5)
    assert [x.title for x in r_work.results] == ["rfc.md"]
    assert r_work.results[0].extra["collection"] == "work"
    assert "rag" in r_work.results[0].extra["tags"]

    r_tag = await eng.search("tag:rag", limit=5)
    assert [x.title for x in r_tag.results] == ["rfc.md"]

    r_col_tag = await eng.search("col:notes tag:architecture hybrid", limit=5)
    assert [x.title for x in r_col_tag.results] == ["journal.md"]


@pytest.mark.asyncio
async def test_scanned_pdf_detection_epub_and_csv_header_chunks(tmp_path):
    """支持扫描件 PDF 显式标记、零依赖 EPUB 章节解析以及 CSV 分块表头保留。"""
    ws = tmp_path / "formats"
    ws.mkdir()
    _make_blank_pdf(ws / "scan_contract.pdf")
    _make_minimal_epub(ws / "handbook.epub")

    csv_lines = ["service_name,region_code,p95_latency_ms"]
    for i in range(1, 56):
        csv_lines.append(f"svc_{i},us-east,{10 + i}")
    csv_lines.append("hyper_search_core,ap-east,12")
    (ws / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    db = tmp_path / "fmt.db"
    stats = index_paths([ws], db_path=db)
    assert stats.indexed == 3

    eng = LocalIndexEngine(db_path=db)

    r_pdf = await eng.search("type:pdf scan_contract", limit=3)
    assert len(r_pdf.results) == 1
    assert r_pdf.results[0].extra.get("scanned_pdf_no_ocr") is True
    assert "scanned-pdf-no-ocr" in r_pdf.results[0].snippet

    r_epub = await eng.search("QuantumLattice", limit=3)
    assert [x.title for x in r_epub.results] == ["handbook.epub"]
    assert r_epub.results[0].extra.get("matched_locator") == "CH1"

    # CSV 深处分块应保留表头列名
    conn = sqlite3.connect(str(db))
    try:
        chunk_texts = [r[0] for r in conn.execute("SELECT text_preview FROM doc_chunks WHERE path LIKE '%metrics.csv'").fetchall()]
    finally:
        conn.close()
    assert len(chunk_texts) >= 2
    assert all("service_name,region_code,p95_latency_ms" in ct for ct in chunk_texts)


@pytest.mark.asyncio
async def test_one_hop_doc_links_graph_for_markdown_and_python(tmp_path):
    """索引期应抽取 Markdown wikilink/相对链接与 Python import，检索时带出库内一跳关联文档。"""
    ws = tmp_path / "graph_ws"
    ws.mkdir()
    (ws / "overview.md").write_text(
        "# System Overview\n\nUniqueOverviewAnchor links to [[design_spec]] and [core engine](engine.py).\n",
        encoding="utf-8",
    )
    (ws / "design_spec.md").write_text("# Design Spec\n\nDetailed architecture specification.\n", encoding="utf-8")
    (ws / "engine.py").write_text(
        "from rank_helper import compute_rrf\n\ndef UniqueEngineAnchor():\n    return compute_rrf()\n",
        encoding="utf-8",
    )
    (ws / "rank_helper.py").write_text("def compute_rrf():\n    return 1.0\n", encoding="utf-8")

    db = tmp_path / "graph.db"
    index_paths([ws], db_path=db)
    eng = LocalIndexEngine(db_path=db)

    r_ov = await eng.search("UniqueOverviewAnchor", limit=3)
    assert r_ov.results[0].title == "overview.md"
    rel_titles = {d["title"] for d in r_ov.results[0].extra.get("related_docs", [])}
    assert "design_spec.md" in rel_titles
    assert "engine.py" in rel_titles

    r_en = await eng.search("UniqueEngineAnchor", limit=3)
    assert r_en.results[0].title == "engine.py"
    rel_py = {d["title"] for d in r_en.results[0].extra.get("related_docs", [])}
    assert "rank_helper.py" in rel_py


@pytest.mark.asyncio
async def test_legacy_db_schema_migration_compatibility(tmp_path):
    """老版本 .omni-index.db（缺新列/新表）升级索引与查询时必须无痛幂等迁移。"""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE docs (
            path TEXT PRIMARY KEY,
            root TEXT NOT NULL,
            title TEXT NOT NULL,
            modality TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            metadata_only INTEGER NOT NULL,
            excerpt TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE docs_fts USING fts5(path UNINDEXED, title_tok, body_tok, tokenize='unicode61');
        INSERT INTO docs VALUES ('/old/doc.md', '/old', 'doc.md', 'document', 20, 1, 0, 'legacy alpha text', '2026-01-01');
        INSERT INTO docs_fts VALUES ('/old/doc.md', 'doc md', 'legacy alpha text');
        """
    )
    conn.commit()
    conn.close()

    ws = tmp_path / "new_ws"
    ws.mkdir()
    (ws / "fresh.md").write_text("# Fresh\n\nNewly added content after migration.\n", encoding="utf-8")

    stats = index_paths([ws], db_path=db, collection="migrated")
    assert stats.indexed == 1

    eng = LocalIndexEngine(db_path=db)
    r_old = await eng.search("legacy alpha", limit=3)
    assert len(r_old.results) == 1
    assert r_old.results[0].title == "doc.md"
    assert r_old.results[0].extra["collection"] == "default"
