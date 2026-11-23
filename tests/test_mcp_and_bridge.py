import json
import pytest

from omni_search.contracts import ReasonCode
from omni_search.mcps_bridge import parse_mcps_json_payload
from omni_search.mcp_server import OmniMCPServer


def test_parse_mcps_json_payload_maps_results_and_degraded_coverage():
    sample = {
        "version": "1.120.0",
        "coverage": {
            "Requested": ["GitHub", "Codeberg"],
            "Executed": ["GitHub"],
            "Blocked": [],
            "Degraded": ["Codeberg"],
            "ReasonCodes": {"Codeberg": "rate-limited"},
            "Verdict": "degraded",
        },
        "warnings": ["[Codeberg] 429 rate-limited"],
        "results": [
            {
                "Platform": "GitHub",
                "Name": "searxng/searxng",
                "Stars": 39773,
                "Unit": "stars",
                "Url": "https://github.com/searxng/searxng",
                "Desc": "Free internet metasearch engine",
                "License": "AGPL-3.0",
                "Updated": "2026-11-22T05:34:12Z",
            }
        ],
    }
    env = parse_mcps_json_payload("mcps_repo", "code", sample, elapsed_ms=420)
    assert env.engine == "mcps_repo"
    assert len(env.results) == 1
    assert env.results[0].title == "searxng/searxng"
    assert env.results[0].popularity == 39773
    assert env.partial is True
    assert env.failure is not None
    assert env.failure.reason_code == ReasonCode.RATE_LIMITED
    # 请求账本恒等式：attempted == succeeded + failed
    assert env.requests.attempted == env.requests.succeeded + env.requests.failed


@pytest.mark.asyncio
async def test_mcp_server_initialize_tools_list_and_doctor_structured_content(tmp_path):
    server = OmniMCPServer(health_path=tmp_path / "health.json")

    init_resp = await server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp["result"]["serverInfo"]["name"] == "search-all-in-one"

    list_resp = await server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = {t["name"]: t for t in list_resp["result"]["tools"]}
    assert {"omni_search", "omni_fetch", "omni_batch", "omni_research", "omni_index", "omni_doctor"}.issubset(tools.keys())
    assert "outputSchema" in tools["omni_search"]
    assert "outputSchema" in tools["omni_research"]
    assert "outputSchema" in tools["omni_index"]

    doc_resp = await server.handle_request(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "omni_doctor", "arguments": {"probe": False}}}
    )
    res = doc_resp["result"]
    assert res["isError"] is False
    assert "structuredContent" in res
    assert res["structuredContent"]["engine_count"] >= 10
    parsed_text = json.loads(res["content"][0]["text"])
    assert parsed_text["engine_count"] == res["structuredContent"]["engine_count"]


@pytest.mark.asyncio
async def test_mcps_bridge_cancel_cleans_subprocess(tmp_path):
    """预算截断触发 asyncio.CancelledError 时必须终止并回收子进程，不留泄漏 transport。"""
    import asyncio
    from omni_search.mcps_bridge import McpsBridgeEngine

    sleeper = tmp_path / "slow_mcps.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    eng = McpsBridgeEngine("mcps_repo", "code", mode="repo", script_path=sleeper, deadline_sec=20.0)
    task = asyncio.create_task(eng.search("cancel-test", limit=3))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 验证内部记录的最近子进程已被终止（returncode 不为 None）
    proc = getattr(eng, "_last_proc", None)
    assert proc is not None
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_mcp_stdio_real_subprocess_handshake_and_doctor():
    """真实 stdio 子进程 JSON-RPC 2.0 握手、6 工具 schema 核验与 omni_doctor 25 引擎自检。"""
    import asyncio
    import sys

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "omni_search.cli",
        "mcp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        reqs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "omni_doctor", "arguments": {"probe": False}}},
        ]
        payload = "".join(json.dumps(r) + "\n" for r in reqs).encode("utf-8")
        out_b, err_b = await asyncio.wait_for(proc.communicate(payload), timeout=15.0)
        assert proc.returncode == 0, err_b.decode("utf-8", errors="replace")
        lines = [ln for ln in out_b.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        assert len(lines) == 3

        r1, r2, r3 = (json.loads(ln) for ln in lines)
        assert r1["result"]["serverInfo"]["name"] == "search-all-in-one"
        tools = {t["name"]: t for t in r2["result"]["tools"]}
        assert len(tools) >= 6
        assert {"omni_search", "omni_fetch", "omni_batch", "omni_research", "omni_index", "omni_doctor"}.issubset(tools)
        research_props = tools["omni_research"]["inputSchema"]["properties"]
        assert {"renderMarkdown", "renderBibtex", "includeLocal", "deepen"}.issubset(research_props.keys())

        doc = r3["result"]["structuredContent"]
        assert doc["engine_count"] == 50
        assert "local_index" in doc
        assert doc["local_index"]["embedder"]
    finally:
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            transport.close()
