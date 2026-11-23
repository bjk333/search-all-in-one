"""跨进程 SQLite WAL 持久化 TTL 查询缓存（仅缓存非空公网结果，绝不缓存故障与本机域）。"""

from __future__ import annotations

import pathlib
import sqlite3
import time


class DiskCacheStore:
    """轻量单文件 SQLite 持久化 TTL 缓存（WAL + busy_timeout 保证多进程并发读写安全）。"""

    def __init__(self, path: pathlib.Path | str, ttl_sec: float = 300.0, max_entries: int = 512):
        self.path = pathlib.Path(path)
        self.ttl_sec = max(0.0, float(ttl_sec))
        self.max_entries = max(32, int(max_entries))
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        if not self._initialized:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    expires_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_cache_exp ON query_cache(expires_at)")
            self._initialized = True
        return conn

    def get(self, key: str) -> str | None:
        if self.ttl_sec <= 0:
            return None
        now = time.time()
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT payload, expires_at FROM query_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if not row:
                    return None
                payload, exp = str(row[0]), float(row[1])
                if exp <= now:
                    conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (key,))
                    conn.commit()
                    return None
                return payload
            finally:
                conn.close()
        except Exception:
            return None

    def set(self, key: str, payload: str, ttl_sec: float | None = None) -> None:
        eff_ttl = self.ttl_sec if ttl_sec is None else max(0.0, float(ttl_sec))
        if eff_ttl <= 0:
            return
        now = time.time()
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache (cache_key, expires_at, payload) VALUES (?, ?, ?)",
                    (key, now + eff_ttl, payload),
                )
                cnt_row = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()
                if cnt_row and int(cnt_row[0]) > self.max_entries:
                    conn.execute("DELETE FROM query_cache WHERE expires_at <= ?", (now,))
                    conn.execute(
                        """
                        DELETE FROM query_cache WHERE cache_key IN (
                            SELECT cache_key FROM query_cache ORDER BY expires_at ASC LIMIT ?
                        )
                        """,
                        (max(1, self.max_entries // 4),),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return
