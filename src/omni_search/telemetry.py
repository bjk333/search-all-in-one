"""telemetry: 基于 SQLite WAL 的本地引擎效费比遥测存储（P50/P95 延迟、成功率、Top-5 贡献与调优建议）。"""

from __future__ import annotations

import pathlib
import sqlite3
import time
from typing import Any


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


class TelemetryStore:
    """轻量 SQLite WAL 遥测库：每次查询批量落盘各源表现，供 `omni doctor --stats` 输出量化面板。"""

    def __init__(self, db_path: pathlib.Path | str, retention_days: int = 30):
        self.db_path = pathlib.Path(db_path)
        self.retention_days = max(1, int(retention_days))
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=3.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=3000;")
        if not self._initialized:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS engine_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    engine TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    reason_code TEXT NOT NULL,
                    elapsed_ms INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    top5_hits INTEGER NOT NULL,
                    skipped_fast INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_engine_events_ts ON engine_events(ts);
                CREATE INDEX IF NOT EXISTS idx_engine_events_eng ON engine_events(engine, ts);
                """
            )
            self._initialized = True
        return conn

    def record_batch(self, events: list[dict[str, Any]], now: float | None = None) -> None:
        if not events:
            return
        ts = time.time() if now is None else now
        rows = [
            (
                ts,
                str(ev.get("engine") or ""),
                str(ev.get("domain") or ""),
                1 if ev.get("ok") else 0,
                str(ev.get("reason_code") or "ok"),
                max(0, int(ev.get("elapsed_ms") or 0)),
                max(0, int(ev.get("result_count") or 0)),
                max(0, int(ev.get("top5_hits") or 0)),
                1 if ev.get("skipped_fast") else 0,
            )
            for ev in events
            if ev.get("engine")
        ]
        if not rows:
            return
        try:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO engine_events
                        (ts, engine, domain, ok, reason_code, elapsed_ms, result_count, top5_hits, skipped_fast)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                cutoff = ts - self.retention_days * 86400.0
                conn.execute("DELETE FROM engine_events WHERE ts < ?", (cutoff,))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def summarize(self, window_days: int = 7, now: float | None = None) -> dict[str, Any]:
        days = max(1, int(window_days))
        ts = time.time() if now is None else now
        since = ts - days * 86400.0

        if not self.db_path.exists():
            return {
                "db_path": str(self.db_path),
                "window_days": days,
                "total_events": 0,
                "engines": [],
                "top_contributors": [],
                "recommendations": [],
            }

        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT engine, domain, ok, reason_code, elapsed_ms, result_count, top5_hits, skipped_fast
                    FROM engine_events
                    WHERE ts >= ?
                    """,
                    (since,),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            rows = []

        grouped: dict[str, list[tuple[Any, ...]]] = {}
        for r in rows:
            grouped.setdefault(str(r[0]), []).append(r)

        engine_stats: list[dict[str, Any]] = []
        recommendations: list[str] = []

        _non_live_reasons = frozenset({"missing-credential", "not-run", "user-disabled"})
        for eng, evs in sorted(grouped.items()):
            calls = len(evs)
            live_evs = [e for e in evs if (not e[7]) and (str(e[3]) not in _non_live_reasons)]
            live_calls = len(live_evs)
            ok_cnt = sum(1 for e in live_evs if e[2])
            fast_skips = sum(1 for e in evs if e[7])
            latencies = sorted(int(e[4]) for e in live_evs)
            p50 = _percentile(latencies, 0.50)
            p95 = _percentile(latencies, 0.95)
            avg_res = round(sum(int(e[5]) for e in live_evs) / live_calls, 2) if live_calls else 0.0
            top5_total = sum(int(e[6]) for e in evs)
            top5_rate = round(top5_total / max(1, live_calls), 3)
            sr = round(ok_cnt / live_calls, 3) if live_calls else 0.0

            fail_reasons: dict[str, int] = {}
            for e in live_evs:
                if not e[2]:
                    rc = str(e[3])
                    fail_reasons[rc] = fail_reasons.get(rc, 0) + 1
            top_fail = max(fail_reasons.items(), key=lambda x: x[1])[0] if fail_reasons else None

            domain = str(evs[-1][1])
            engine_stats.append(
                {
                    "engine": eng,
                    "domain": domain,
                    "calls": calls,
                    "live_calls": live_calls,
                    "fast_skips": fast_skips,
                    "success_rate": sr,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "avg_results": avg_res,
                    "top5_contribution_total": top5_total,
                    "top5_rate": top5_rate,
                    "top_failure_reason": top_fail,
                }
            )

            if live_calls >= 3 and sr < 0.30:
                recommendations.append(
                    f"[{eng}] 近 {days} 天实测成功率仅 {sr:.0%}（主因={top_fail}，P95={p95}ms），建议核查网络/凭据或加入 OMNI_DISABLE"
                )
            elif live_calls >= 5 and sr >= 0.80 and avg_res >= 3.0 and top5_total == 0:
                recommendations.append(
                    f"[{eng}] 疑似高产出零贡献/假活源（成功率 {sr:.0%}、均产 {avg_res} 条但 Top-5 贡献=0），建议核查上游查询参数或映射质量"
                )
            elif live_calls >= 3 and p95 >= 6000 and top5_total == 0:
                recommendations.append(
                    f"[{eng}] 高延迟低贡献（P95={p95}ms，Top-5 贡献=0），建议收紧单源 deadline 或按需禁用"
                )

        engine_stats.sort(key=lambda x: (x["top5_contribution_total"], x["success_rate"], -x["p95_ms"]), reverse=True)
        top_contributors = [
            {"engine": e["engine"], "domain": e["domain"], "top5_hits": e["top5_contribution_total"], "p50_ms": e["p50_ms"]}
            for e in engine_stats[:8]
            if e["top5_contribution_total"] > 0
        ]

        return {
            "db_path": str(self.db_path),
            "window_days": days,
            "total_events": len(rows),
            "engines": engine_stats,
            "top_contributors": top_contributors,
            "recommendations": recommendations,
        }
