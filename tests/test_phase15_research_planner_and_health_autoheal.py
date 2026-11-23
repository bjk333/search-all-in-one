"""阶段 15 单测：research_planner 模块拆分瘦身（全仓 < 700 行护城河）与
凭据指纹感知的熔断自动自愈（Credential-Fingerprint Auto-Heal）。"""

from __future__ import annotations

import pathlib

import pytest

from omni_search.contracts import EngineEnvelope, EngineFailure, ReasonCode, RequestStats, SearchItem
from omni_search.declarative import DeclarativeEngine, load_declarative_engines
from omni_search.health import HealthStore
from omni_search.pool import FederatedPool

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_research_planner_split_and_global_module_line_budget():
    """research_planner.py 拆出后，research.py 应降至 < 680 行，且全包所有模块 < 700 行。"""
    planner_file = PROJECT_ROOT / "src" / "omni_search" / "research_planner.py"
    research_file = PROJECT_ROOT / "src" / "omni_search" / "research.py"
    assert planner_file.exists(), "research_planner.py must exist"

    research_lines = len(research_file.read_text(encoding="utf-8").splitlines())
    assert research_lines < 680, f"research.py should be < 680 lines after split, got {research_lines}"

    for py_file in (PROJECT_ROOT / "src" / "omni_search").glob("*.py"):
        cnt = len(py_file.read_text(encoding="utf-8").splitlines())
        assert cnt < 700, f"{py_file.name} exceeds 700-line budget: {cnt} lines"


def test_research_module_reexports_planner_symbols():
    """research.py 必须向后兼容 re-export 规划器核心符号。"""
    from omni_search import research, research_planner

    for name in (
        "widen_query",
        "plan_research_lanes",
        "extract_followup_terms",
        "_compute_query_relevance",
        "_is_cjk_dominant",
    ):
        assert getattr(research, name) is getattr(research_planner, name)


def test_health_store_auto_heals_when_credential_fingerprint_changes(tmp_path):
    """引擎因 forbidden/quota-exhausted/rate-limited 熔断后，一旦凭据指纹变化应自动解封且不落盘明文密钥。"""
    hpath = tmp_path / "health.json"
    store = HealthStore(hpath)
    secret_old = "ghp_old_secret_value_12345"
    secret_new = "ghp_new_secret_value_67890"

    from omni_search.health import make_credential_fingerprint

    fp_old = make_credential_fingerprint(secret_old)
    fp_new = make_credential_fingerprint(secret_new)
    assert fp_old and fp_new and fp_old != fp_new
    assert len(fp_old) == 12
    assert secret_old not in fp_old

    # 无凭据时触发 forbidden 持久熔断
    store.record_failure("github_direct", ReasonCode.FORBIDDEN, now=1000.0, cred_fingerprint="")
    st_locked = store.check("github_direct", now=1010.0, cred_fingerprint="")
    assert st_locked.in_cooldown is True
    assert st_locked.skip_fast is True
    assert st_locked.auto_healed is False

    # 配上新凭据后，check 应自动解封自愈
    st_healed = store.check("github_direct", now=1020.0, cred_fingerprint=fp_old)
    assert st_healed.in_cooldown is False
    assert st_healed.skip_fast is False
    assert st_healed.consecutive_failures == 0
    assert st_healed.auto_healed is True

    # 用旧凭据触发 quota_exhausted 后，同一凭据仍保持熔断
    store.record_failure("deepseek", ReasonCode.QUOTA_EXHAUSTED, now=2000.0, cred_fingerprint=fp_old)
    raw_disk = hpath.read_text(encoding="utf-8")
    assert secret_old not in raw_disk  # 绝不落盘明文密钥
    assert store.check("deepseek", now=2010.0, cred_fingerprint=fp_old).skip_fast is True

    # 更换新凭据后立即自动自愈
    st_ds_healed = store.check("deepseek", now=2020.0, cred_fingerprint=fp_new)
    assert st_ds_healed.skip_fast is False
    assert st_ds_healed.auto_healed is True

    # 瞬时网络 deadline 不因凭据变化误解封（仅凭据相关原因自愈）
    for i in range(4):
        store.record_failure("arxiv", ReasonCode.DEADLINE, now=3000.0 + i, cred_fingerprint=fp_old)
    st_net = store.check("arxiv", now=3010.0, cred_fingerprint=fp_new)
    assert st_net.in_cooldown is True
    assert st_net.auto_healed is False


@pytest.mark.asyncio
async def test_pool_auto_heals_engine_after_token_added(tmp_path):
    """FederatedPool 在引擎新增有效凭据后应自动解除历史 forbidden 熔断并正常执行请求。"""

    class _TokenAwareEngine:
        id = "github_direct"
        domain = "code"
        tier = "L0"
        deadline_sec = 5.0

        def __init__(self, token: str):
            self.token = token
            self.calls = 0

        def credential_status(self):
            return True, None

        def credential_fingerprint(self) -> str:
            from omni_search.health import make_credential_fingerprint

            return make_credential_fingerprint(self.token)

        async def search(self, query: str, limit: int, client) -> EngineEnvelope:
            self.calls += 1
            if not self.token:
                return EngineEnvelope(
                    engine=self.id,
                    domain=self.domain,
                    elapsed_ms=5,
                    requests=RequestStats(attempted=1, succeeded=0, failed=1),
                    failure=EngineFailure(reason_code=ReasonCode.FORBIDDEN, reason="403 anonymous"),
                )
            return EngineEnvelope(
                engine=self.id,
                domain=self.domain,
                elapsed_ms=5,
                requests=RequestStats(attempted=1, succeeded=1, failed=0),
                results=[SearchItem(id="r1", title="Omni Search Repo", url="https://github.com/org/omni-search", source=self.id, domain=self.domain)],
            )

    hstore = HealthStore(tmp_path / "h.json")
    eng_anon = _TokenAwareEngine("")
    pool1 = FederatedPool(engines=[eng_anon], health_store=hstore, disk_cache=None)
    try:
        r1 = await pool1.search("omni", domains=["code"], limit=3)
        assert len(r1.results) == 0
        assert hstore.check("github_direct").skip_fast is True
    finally:
        await pool1.aclose()

    # 第二次带上新 GITHUB_TOKEN，无需手动 reset 即自动自愈并命中结果
    eng_authed = _TokenAwareEngine("ghp_fresh_token_xyz")
    pool2 = FederatedPool(engines=[eng_authed], health_store=hstore, disk_cache=None)
    try:
        r2 = await pool2.search("omni", domains=["code"], limit=3)
        assert eng_authed.calls == 1
        assert len(r2.results) == 1
        assert hstore.check("github_direct", cred_fingerprint=eng_authed.credential_fingerprint()).in_cooldown is False
    finally:
        await pool2.aclose()


def test_declarative_engine_exposes_credential_fingerprint():
    """DeclarativeEngine 应根据其 auth.env / url_env 生成确定性短指纹且不暴露明文。"""
    specs = load_declarative_engines(PROJECT_ROOT / "engines")
    gh_spec = next(s for s in specs if s.id == "github_direct")
    e_empty = DeclarativeEngine(gh_spec, env={})
    e_tok = DeclarativeEngine(gh_spec, env={"GITHUB_TOKEN": "ghp_secret_abc"})
    assert e_empty.credential_fingerprint() == ""
    fp = e_tok.credential_fingerprint()
    assert len(fp) == 12
    assert "ghp_secret_abc" not in fp
