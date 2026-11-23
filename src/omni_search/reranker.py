"""可选跨源精排器（兼容 Cohere / Jina / SiliconFlow / vLLM 标准 /v1/rerank 协议）。"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from omni_search.contracts import SearchItem


class Reranker(Protocol):
    id: str

    async def rerank(
        self,
        query: str,
        items: list[SearchItem],
        client: httpx.AsyncClient,
        top_n: int = 20,
    ) -> list[SearchItem]: ...


class HttpReranker:
    """标准 HTTP JSON `/v1/rerank` 精排器（默认未配置时不启用，失败由调度池优雅降级回 RRF）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_sec: float = 8.0,
    ):
        clean_url = base_url.strip().rstrip("/")
        if not clean_url.endswith("/rerank"):
            clean_url = f"{clean_url}/rerank"
        self.endpoint = clean_url
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_sec = float(timeout_sec)
        self.id = f"http_rerank:{self.model}"

    async def rerank(
        self,
        query: str,
        items: list[SearchItem],
        client: httpx.AsyncClient,
        top_n: int = 20,
    ) -> list[SearchItem]:
        if len(items) <= 1:
            return items
        window = items[: max(2, min(len(items), top_n))]
        tail = items[len(window) :]
        docs = [f"{it.title}\n{it.snippet}".strip()[:1200] for it in window]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": len(window),
        }
        resp = await client.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results") or []
        if not isinstance(raw_results, list) or not raw_results:
            return items

        scored: list[tuple[float, int]] = []
        seen_idx: set[int] = set()
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            score = entry.get("relevance_score", entry.get("score"))
            if isinstance(idx, int) and 0 <= idx < len(window) and idx not in seen_idx:
                seen_idx.add(idx)
                try:
                    num_score = float(score)
                except (TypeError, ValueError):
                    num_score = 0.0
                scored.append((num_score, idx))

        if not scored:
            return items

        scored.sort(key=lambda x: x[0], reverse=True)
        reordered: list[SearchItem] = []
        for num_score, idx in scored:
            copied = window[idx].model_copy(deep=True)
            copied.extra["rerank_score"] = round(num_score, 5)
            copied.extra["reranker"] = self.id
            reordered.append(copied)
        for idx, it in enumerate(window):
            if idx not in seen_idx:
                reordered.append(it)
        reordered.extend(tail)
        return reordered


def build_default_reranker(env: dict[str, str] | None = None) -> Reranker | None:
    """配置了 OMNI_RERANK_URL + OMNI_RERANK_MODEL 时启用标准 /v1/rerank 精排，否则返回 None（零开销）。"""
    eff = os.environ if env is None else env
    url = eff.get("OMNI_RERANK_URL", "").strip()
    model = eff.get("OMNI_RERANK_MODEL", "").strip()
    if not (url and model):
        return None
    api_key = eff.get("OMNI_RERANK_API_KEY", "").strip()
    timeout_sec = float(eff.get("OMNI_RERANK_TIMEOUT", "8.0") or "8.0")
    return HttpReranker(base_url=url, model=model, api_key=api_key, timeout_sec=timeout_sec)
