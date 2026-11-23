"""阶段 12A–12B 回归测试：GitHub 免配置凭据解析（GITHUB_TOKEN/GH_TOKEN/gh auth token）与 Agentic Web 响应头感知（Content-Signal / x-markdown-tokens）。"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import httpx
import pytest

from omni_search import contracts, fetcher
from omni_search.mcp_server import load_dotenv_into_env


def test_resolve_github_token_precedence_and_gh_cli_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1) GITHUB_TOKEN 优先于 GH_TOKEN
    assert (
        contracts.resolve_github_token({"GITHUB_TOKEN": "tok_primary", "GH_TOKEN": "tok_gh"}, allow_gh_cli=False)
        == "tok_primary"
    )
    # 2) 无 GITHUB_TOKEN 时自动复用 GH_TOKEN
    assert contracts.resolve_github_token({"GH_TOKEN": "tok_gh"}, allow_gh_cli=False) == "tok_gh"
    # 3) OMNI_NO_GH_CLI_TOKEN=1 禁用 gh CLI 回退
    assert contracts.resolve_github_token({"OMNI_NO_GH_CLI_TOKEN": "1"}, allow_gh_cli=True) == ""

    # 4) 无环境变量时通过 gh auth token 回退并进程内缓存
    monkeypatch.setattr(contracts, "_GH_CLI_TOKEN_CACHE", "")
    monkeypatch.setattr(contracts, "_GH_CLI_TOKEN_PROBED", False)
    calls = {"n": 0}

    def fake_which(name: str) -> str | None:
        return "/usr/bin/gh" if name == "gh" else None

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stdout="gho_fake_cli_token_123\n")

    import shutil

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert contracts.resolve_github_token({}, allow_gh_cli=True) == "gho_fake_cli_token_123"
    assert contracts.resolve_github_token({}, allow_gh_cli=True) == "gho_fake_cli_token_123"
    assert calls["n"] == 1  # 进程内缓存仅调用一次


def test_load_dotenv_bridges_gh_token_into_github_token(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_from_gh_env")
    monkeypatch.setenv("OMNI_NO_MCPS_ENV", "1")
    merged = load_dotenv_into_env(tmp_path / "nonexistent.env")
    assert merged.get("GITHUB_TOKEN") == "ghp_from_gh_env"


@pytest.mark.asyncio
async def test_fetch_parses_content_signal_and_markdown_tokens_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://cf-agent.example.com/docs":
            return httpx.Response(
                200,
                text="# Cloudflare Markdown for Agents\n\nClean markdown body.",
                headers={
                    "content-type": "text/markdown; charset=utf-8",
                    "content-signal": "search=yes, ai-train=no, ai-input=yes",
                    "x-markdown-tokens": "1842",
                },
            )
        if url == "https://cf-agent.example.com/bad-header":
            return httpx.Response(
                200,
                text="# Bad Header Test\n\nStill succeeds cleanly.",
                headers={
                    "content-type": "text/markdown",
                    "content-signal": "malformed_directive, search=yes",
                    "x-markdown-tokens": "not-a-number",
                },
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as cli:
        r1 = await fetcher.safe_fetch_url(
            "https://cf-agent.example.com/docs",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r1.ok is True
        assert r1.source_via == "markdown-negotiated"
        assert r1.title == "Cloudflare Markdown for Agents"
        assert r1.content_signals == {"search": "yes", "ai-train": "no", "ai-input": "yes"}
        assert r1.markdown_tokens == 1842

        r2 = await fetcher.safe_fetch_url(
            "https://cf-agent.example.com/bad-header",
            client=cli,
            resolve_host=False,
            use_cache=False,
        )
        assert r2.ok is True
        assert r2.content_signals == {"search": "yes"}
        assert r2.markdown_tokens is None
