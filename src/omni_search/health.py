"""自适应健康记忆与半开熔断（借鉴 multi-code-platform-searcher 降级治理）。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
import time
from typing import Any

from omni_search.contracts import ReasonCode


_PERSISTENT_COOLDOWNS: dict[ReasonCode, float] = {
    ReasonCode.QUOTA_EXHAUSTED: 6 * 3600.0,  # 余额耗尽：6 小时 Fast-Fail
    ReasonCode.FORBIDDEN: 30 * 60.0,         # 403 拒绝：30 分钟 Fast-Fail
}
_CREDENTIAL_HEALABLE_REASONS = frozenset({
    ReasonCode.FORBIDDEN.value,
    ReasonCode.QUOTA_EXHAUSTED.value,
    ReasonCode.RATE_LIMITED.value,
})


def make_credential_fingerprint(*values: str) -> str:
    """对引擎凭据值生成 12 位不可逆 SHA-256 短指纹（绝不落盘明文密钥）；无凭据返回空串。"""
    non_empty = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not non_empty:
        return ""
    raw = "\x1f".join(non_empty).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


@dataclass
class HealthStatus:
    in_cooldown: bool
    consecutive_failures: int
    last_reason: str | None
    cooldown_until: float
    probe_deadline_sec: float
    skip_fast: bool = False
    half_open: bool = False
    auto_healed: bool = False


class HealthStore:
    """跨调用记录引擎连续失败，冷却期内收紧单引擎探测截止时间避免拖垮尾延迟。"""

    def __init__(self, path: pathlib.Path | str | None = None, probe_deadline_sec: float = 5.0):
        self.path = pathlib.Path(path) if path else None
        self.probe_deadline_sec = probe_deadline_sec
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = raw
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def check(self, engine_id: str, now: float | None = None, cred_fingerprint: str = "") -> HealthStatus:
        ts = time.time() if now is None else now
        rec = self._data.get(engine_id, {})
        fails = int(rec.get("consecutive_failures", 0))
        until = float(rec.get("cooldown_until", 0.0))
        last_reason = rec.get("last_reason")
        is_persistent = last_reason in (ReasonCode.QUOTA_EXHAUSTED.value, ReasonCode.FORBIDDEN.value)
        had_tripped = fails >= 2 or (is_persistent and fails >= 1)
        prev_fp = str(rec.get("cred_fingerprint") or "")
        # 凭据感知自动自愈：若引擎因 forbidden/quota/rate-limited 处于熔断态，且检测到新凭据指纹，自动解封
        if had_tripped and last_reason in _CREDENTIAL_HEALABLE_REASONS and cred_fingerprint and cred_fingerprint != prev_fp:
            self._data.pop(engine_id, None)
            self._save()
            return HealthStatus(
                in_cooldown=False,
                consecutive_failures=0,
                last_reason=None,
                cooldown_until=0.0,
                probe_deadline_sec=0.0,
                skip_fast=False,
                half_open=False,
                auto_healed=True,
            )
        in_cd = ts < until and had_tripped
        # 冷却到期但尚未探活成功：进入 half_open 状态，仍收紧 probe_deadline_sec 防尾延迟拖垮
        half_open = bool((not in_cd) and had_tripped)
        # 持久性故障或连续 >=4 次失败在冷却期内直接 Fast-Fail 跳过网络请求；2-3 次失败保持半开紧缩探测
        skip_fast = bool(in_cd and (is_persistent or fails >= 4))
        return HealthStatus(
            in_cooldown=in_cd,
            consecutive_failures=fails,
            last_reason=last_reason,
            cooldown_until=until,
            probe_deadline_sec=self.probe_deadline_sec if (in_cd or half_open) else 0.0,
            skip_fast=skip_fast,
            half_open=half_open,
        )

    def reset(self, engine_id: str | None = None) -> None:
        """手动或凭据更新后清零全部或单个引擎的熔断记录。"""
        if engine_id is None:
            self._data.clear()
        else:
            self._data.pop(engine_id, None)
        self._save()

    def record_success(self, engine_id: str, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        if engine_id in self._data:
            self._data[engine_id] = {
                "consecutive_failures": 0,
                "last_reason": None,
                "cooldown_until": 0.0,
                "last_ok_at": ts,
            }
            self._save()

    def record_failure(
        self,
        engine_id: str,
        reason_code: ReasonCode,
        now: float | None = None,
        cred_fingerprint: str = "",
    ) -> None:
        # 缺凭据/用户禁用属于静态配置态，不计入瞬时网络熔断
        if reason_code in (ReasonCode.MISSING_CREDENTIAL, ReasonCode.USER_DISABLED, ReasonCode.ZERO_HIT):
            return
        ts = time.time() if now is None else now
        rec = self._data.get(engine_id, {})
        fails = int(rec.get("consecutive_failures", 0)) + 1
        if reason_code in _PERSISTENT_COOLDOWNS:
            cooldown_sec = _PERSISTENT_COOLDOWNS[reason_code]
        else:
            # 指数退避冷却：连续 2 次失败 -> 60s，连续 >=4 次进入 Fast-Fail 最高 1800s（30m）
            cooldown_sec = min(1800.0, 30.0 * (2 ** max(0, fails - 1))) if fails >= 2 else 0.0
        self._data[engine_id] = {
            "consecutive_failures": fails,
            "last_reason": reason_code.value,
            "cooldown_until": ts + cooldown_sec,
            "last_fail_at": ts,
            "cred_fingerprint": cred_fingerprint,
        }
        self._save()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)
