"""可插拔本地/兼容向量嵌入引擎（默认零外部依赖 HashNgram，可选 Ollama / OpenAI /v1/embeddings）。"""

from __future__ import annotations

from array import array
import hashlib
import math
import os
import re
from typing import Protocol

import httpx

_ASCII_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_CJK_SEG_RE = re.compile(r"[\u4e00-\u9fff]+")


class Embedder(Protocol):
    id: str
    dim: int
    min_score: float

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    inv = 1.0 / norm
    return [x * inv for x in vec]


def pack_vector(vec: list[float]) -> bytes:
    """将 float32 向量打包为紧凑小端 BLOB（256 维仅 1 KiB）。"""
    arr = array("f", vec)
    if arr.itemsize != 4:
        arr = array("f", [float(x) for x in vec])
    return arr.tobytes()


def unpack_vector(blob: bytes) -> array:
    arr = array("f")
    arr.frombytes(blob)
    return arr


def dot_similarity(a: list[float] | array, b: list[float] | array) -> float:
    """L2 归一化向量点积即余弦相似度。"""
    if len(a) != len(b) or not a:
        return 0.0
    return float(math.fsum(x * y for x, y in zip(a, b)))


def _extract_subword_features(text: str) -> list[tuple[str, float]]:
    """提取词级 + 字符 3-gram + 中文 2-gram 特征，兼顾精确词形与近似拼写/词干重叠。"""
    low = text.lower()
    feats: list[tuple[str, float]] = []

    for m in _ASCII_WORD_RE.finditer(low):
        w = m.group(0)
        feats.append((f"w:{w}", 1.5))
        padded = f"^{w}$"
        if len(padded) >= 3:
            for i in range(len(padded) - 2):
                feats.append((f"g:{padded[i : i + 3]}", 0.6))

    for m in _CJK_SEG_RE.finditer(text):
        seg = m.group(0)
        if len(seg) == 1:
            feats.append((f"c1:{seg}", 1.2))
        else:
            for ch in seg:
                feats.append((f"c1:{ch}", 0.5))
            for i in range(len(seg) - 1):
                feats.append((f"c2:{seg[i : i + 2]}", 1.4))
    return feats


class HashNgramEmbedder:
    """零外部依赖、跨进程确定性的子词哈希投影嵌入器（默认 256 维 float32）。"""

    def __init__(self, dim: int = 256, min_score: float = 0.25):
        self.dim = max(32, int(dim))
        self.min_score = float(min_score)
        self.id = f"hash_ngram_v1_{self.dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat, weight in _extract_subword_features(text[:32000]):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=4).digest()
            h = int.from_bytes(digest, "little")
            idx = (h >> 1) % self.dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign * weight
        return l2_normalize(vec)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t or "") for t in texts]


class OpenAICompatEmbedder:
    """兼容本地 Ollama (/v1/embeddings) 与任意 OpenAI Embeddings API 的稠密语义嵌入器。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_sec: float = 12.0,
        min_score: float = 0.35,
    ):
        clean_url = base_url.strip().rstrip("/")
        if not clean_url.endswith("/embeddings"):
            clean_url = f"{clean_url}/embeddings"
        self.endpoint = clean_url
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_sec = float(timeout_sec)
        self.min_score = float(min_score)
        self.dim = 0
        self.id = f"openai_compat:{self.model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": [t[:8000] if t else " " for t in texts]}
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(data.get("data") or [], key=lambda x: int(x.get("index", 0)))
        out = [l2_normalize([float(v) for v in (item.get("embedding") or [])]) for item in items]
        if out and out[0]:
            self.dim = len(out[0])
        return out


class Model2VecEmbedder:
    """可选纯 CPU 静态词向量嵌入器（MinishLab/model2vec，比 Transformer 快 50–500 倍，零 GPU 需求）。"""

    def __init__(
        self,
        model_name: str = "minishlab/potion-multilingual-128M",
        min_score: float = 0.35,
        model: object | None = None,
    ):
        self.model_name = model_name.strip()
        self.min_score = float(min_score)
        self.dim = 0
        self.id = f"model2vec:{self.model_name}"
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            try:
                from model2vec import StaticModel  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "model2vec not installed; run `uv sync --extra semantic` or unset OMNI_EMBED_BACKEND"
                ) from exc
            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        mdl = self._ensure_model()
        raw = mdl.encode([t[:8000] if t else " " for t in texts])
        out: list[list[float]] = []
        for row in raw:
            vec = l2_normalize([float(v) for v in row])
            out.append(vec)
        if out and out[0]:
            self.dim = len(out[0])
        return out


def build_default_embedder(env: dict[str, str] | None = None) -> Embedder:
    """按环境变量自动选择嵌入器：model2vec -> OpenAI/Ollama 兼容端点 -> 默认零依赖 HashNgram。"""
    eff = os.environ if env is None else env
    backend = eff.get("OMNI_EMBED_BACKEND", "").strip().lower()
    if backend == "model2vec":
        model = eff.get("OMNI_EMBED_MODEL", "").strip() or "minishlab/potion-multilingual-128M"
        min_score = float(eff.get("OMNI_EMBED_MIN_SCORE", "0.35") or "0.35")
        return Model2VecEmbedder(model_name=model, min_score=min_score)
    url = eff.get("OMNI_EMBED_URL", "").strip()
    model = eff.get("OMNI_EMBED_MODEL", "").strip()
    if url and model:
        api_key = eff.get("OMNI_EMBED_API_KEY", "").strip()
        min_score = float(eff.get("OMNI_EMBED_MIN_SCORE", "0.35") or "0.35")
        return OpenAICompatEmbedder(base_url=url, model=model, api_key=api_key, min_score=min_score)
    return HashNgramEmbedder()
