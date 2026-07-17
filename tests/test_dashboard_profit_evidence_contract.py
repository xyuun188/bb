from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from web_dashboard.api import dashboard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
HTML = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
STYLE = (ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")


def test_position_risk_evidence_preserves_authoritative_contract_values() -> None:
    decision = SimpleNamespace(
        id=731,
        raw_llm_response={
            "profit_risk_sizing": {
                "contract_version": "dynamic-risk-v1",
                "production_eligible": True,
                "reason": "independent_dynamic_risk_budget_ready",
                "risk_budget_usdt": 4.2,
                "portfolio_risk_budget_usdt": 9.1,
                "remaining_portfolio_risk_budget_usdt": 5.6,
                "current_portfolio_stressed_loss_usdt": 3.5,
                "planned_stressed_loss_usdt": 3.0,
                "stressed_loss_fraction": 0.03,
                "target_notional_usdt": 120.0,
                "final_notional_usdt": 100.0,
                "expected_net_return_pct": 0.8,
                "portfolio_risk_snapshot": {
                    "gross_notional_usdt": 340.0,
                    "same_side_notional_usdt": 210.0,
                    "direction_concentration": 0.617647,
                },
                "execution_reconciliations": [
                    {
                        "source": "okx_exchange_precision",
                        "final_notional_usdt": 100.0,
                        "final_leverage": 3.0,
                        "eligible": True,
                        "reasons": [],
                    }
                ],
                "policy_provenance": {
                    "source": "independent_return_budget",
                    "observation_window": "current_decision",
                    "sample_count": 3,
                    "generated_at": "2026-07-15T00:00:00+00:00",
                    "strategy_version": "dynamic-risk-v1",
                    "fallback_reason": "",
                    "contract_fingerprint": "abc123",
                },
            }
        },
    )

    contract = dashboard._dashboard_position_risk_contract(decision)

    assert contract["available"] is True
    assert contract["risk_budget_usdt"] == 4.2
    assert contract["planned_stressed_loss_usdt"] == 3.0
    assert contract["target_notional_usdt"] == 120.0
    assert contract["final_notional_usdt"] == 100.0
    assert contract["portfolio_risk_snapshot"]["gross_notional_usdt"] == 340.0
    assert contract["adjustment_reasons"] == [
        "final_notional_reduced_from_dynamic_target",
        "okx_exchange_precision",
    ]
    assert contract["policy_provenance"]["contract_fingerprint"] == "abc123"

    incomplete = dashboard._dashboard_position_risk_contract(
        SimpleNamespace(
            id=733,
            raw_llm_response={
                "profit_risk_sizing": {
                    "production_eligible": False,
                    "final_notional_usdt": 30.0,
                }
            },
        )
    )
    assert incomplete["available"] is False
    assert "profit_risk_sizing_not_production_eligible" in incomplete["blockers"]
    assert "independent_risk_budget_missing" in incomplete["evidence_gaps"]
    assert "planned_stressed_loss_missing" in incomplete["evidence_gaps"]
    assert "risk_contract_fingerprint_missing" in incomplete["evidence_gaps"]


def test_missing_position_risk_contract_is_not_rendered_as_zero() -> None:
    contract = dashboard._dashboard_position_risk_contract(
        SimpleNamespace(id=732, raw_llm_response={})
    )

    assert contract == {
        "available": False,
        "decision_id": 732,
        "blockers": ["profit_risk_sizing_missing"],
    }


def test_current_management_archives_legacy_entry_evidence_gaps() -> None:
    envelope = dashboard._dashboard_position_risk_envelope(
        {
            "current_management_contract": {
                "contract_version": "2026-07-15.current-position-management.v1",
                "management_eligible": True,
                "blockers": [],
            }
        },
        contracts=[],
        blockers=["entry_decision_risk_contract_missing"],
    )

    assert envelope["available"] is False
    assert envelope["effective_available"] is True
    assert envelope["current_management_authoritative"] is True
    assert envelope["historical_entry_incomplete"] is True
    assert envelope["blockers"] == []
    assert envelope["historical_blockers"] == ["entry_decision_risk_contract_missing"]
    assert "旧版已归档" in SCRIPT
    assert "旧版入场合同缺口已保留" in SCRIPT
    assert "OKX 原生保护单（无需本地关联）" in SCRIPT


@pytest.mark.asyncio
async def test_split_dashboard_uses_lightweight_executor_for_oco_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    class LightweightExecutor:
        def __init__(self, *, mode: str, load_markets_on_initialize: bool) -> None:
            assert mode == "paper"
            assert load_markets_on_initialize is False

        async def initialize(self) -> None:
            lifecycle.append("initialize")

        async def shutdown(self) -> None:
            lifecycle.append("shutdown")

        async def get_positions_strict(self) -> list[dict]:
            return [
                {
                    "symbol": "EDGE/USDT",
                    "side": "long",
                    "contracts": "2",
                    "info": {"instId": "EDGE-USDT-SWAP", "pos": "2"},
                }
            ]

        async def get_position_protection_orders(self) -> list[dict]:
            return [
                {
                    "symbol": "EDGE/USDT",
                    "position_side": "long",
                    "algo_id": "oco-edge-fallback",
                    "contracts": "2",
                    "reduce_only": True,
                    "state": "live",
                    "order_type": "oco",
                    "stop_loss_price": 96.0,
                    "take_profit_price": 112.0,
                    "created_at_ms": 1,
                }
            ]

        async def get_open_orders_strict(self) -> list[dict]:
            return []

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "OKXExecutor", LightweightExecutor)
    rows = [{"symbol": "EDGE/USDT", "side": "long"}]

    inventory = await dashboard._dashboard_open_position_protection_evidence(
        rows,
        mode="paper",
    )

    assert lifecycle == ["initialize", "shutdown"]
    assert inventory["available"] is True
    assert inventory["blockers"] == []
    assert rows[0]["protection_contract"]["available"] is True
    assert rows[0]["protection_contract"]["orders"][0]["algo_id"] == (
        "oco-edge-fallback"
    )


def test_prediction_economics_reads_persisted_distribution_and_cost_once() -> None:
    raw = {
        "opportunity_score": {
            "production_eligible": True,
            "side": "short",
            "return_distribution_contract": {
                "raw_expected_return_pct": 0.9,
                "objective_expected_return_pct": 0.4,
                "blockers": [],
            },
            "execution_cost": {
                "fee_pct": 0.1,
                "slippage_pct": 0.2,
                "total_pct": 0.3,
                "production_eligible": True,
                "reason": "live_order_size_microstructure_cost_ready",
            },
            "expected_net_breakdown": {
                "net_pct": 0.4,
                "model_gross_pct": 0.9,
                "live_execution_cost_pct": 0.3,
                "authoritative_slippage_tail_excess_pct": 0.2,
                "cost_deduction_count": 1,
            },
        },
        "production_return_policy": {
            "eligible": True,
            "expected_net_return_pct": 0.4,
            "return_lcb_pct": 0.2,
        },
    }

    evidence = dashboard._display_prediction_economics(raw)

    assert evidence["production_eligible"] is True
    assert evidence["return_distribution_contract"]["raw_expected_return_pct"] == 0.9
    assert evidence["execution_cost"]["fee_pct"] == 0.1
    assert evidence["execution_cost"]["slippage_pct"] == 0.2
    assert evidence["cost_and_return_breakdown"]["cost_deduction_count"] == 1
    assert evidence["blockers"] == []

    incomplete = dashboard._display_prediction_economics(
        {"opportunity_score": {"production_eligible": True}}
    )
    assert incomplete["available"] is False
    assert incomplete["blockers"] == [
        "production_return_distribution_missing",
        "live_execution_cost_missing",
        "expected_net_breakdown_missing",
        "production_return_policy_missing",
    ]


@pytest.mark.asyncio
async def test_open_positions_endpoint_returns_risk_and_oco_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'dashboard-profit-evidence.db').as_posix()}",
    )
    await init_db()

    exchange_position = {
        "symbol": "EDGE/USDT",
        "side": "long",
        "contracts": "2",
        "info": {"instId": "EDGE-USDT-SWAP", "pos": "2"},
    }
    exchange_mark = {
        "quantity": 2.0,
        "entry_price": 100.0,
        "mark_price": 103.0,
        "unrealized_pnl": 6.0,
        "margin": 50.0,
    }
    protection_order = {
        "symbol": "EDGE/USDT",
        "position_side": "long",
        "algo_id": "oco-edge-1",
        "contracts": "2",
        "reduce_only": True,
        "state": "live",
        "order_type": "oco",
        "stop_loss_price": 96.0,
        "take_profit_price": 112.0,
        "created_at_ms": 1,
    }

    class FakeExecutor:
        async def get_position_protection_orders(self):
            return [protection_order]

        async def get_open_orders_strict(self):
            return []

    async def exchange_marks(_mode: str | None = None):
        return {("EDGE/USDT", "long"): exchange_mark}

    async def open_symbols(_mode: str | None = None):
        return {"EDGE/USDT"}

    async def public_tickers(_symbols: set[str]):
        return {}

    async def okx_positions(_mode: str):
        return [exchange_position]

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", exchange_marks)
    monkeypatch.setattr(dashboard, "_get_display_open_position_symbols", open_symbols)
    monkeypatch.setattr(dashboard, "_get_public_ticker_map", public_tickers)
    monkeypatch.setattr(dashboard, "_fetch_dashboard_okx_positions", okx_positions)
    monkeypatch.setattr(
        dashboard,
        "_dashboard_okx_executor_for_mode",
        lambda _mode: FakeExecutor(),
    )

    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="EDGE/USDT",
                action="long",
                confidence=0.7,
                position_size_pct=0.1,
                suggested_leverage=2.0,
                stop_loss_pct=0.04,
                take_profit_pct=0.12,
                raw_llm_response={
                    "profit_risk_sizing": {
                        "contract_version": "dynamic-risk-v1",
                        "production_eligible": True,
                        "reason": "independent_dynamic_risk_budget_ready",
                        "risk_budget_usdt": 4.0,
                        "planned_stressed_loss_usdt": 3.2,
                        "current_portfolio_stressed_loss_usdt": 3.5,
                        "target_notional_usdt": 220.0,
                        "final_notional_usdt": 200.0,
                        "portfolio_risk_snapshot": {
                            "gross_notional_usdt": 200.0,
                            "same_side_notional_usdt": 200.0,
                            "direction_concentration": 1.0,
                        },
                        "policy_provenance": {
                            "source": "independent_return_budget",
                            "observation_window": "current_decision",
                            "sample_count": 5,
                            "generated_at": "2026-07-15T00:00:00+00:00",
                            "strategy_version": "dynamic-risk-v1",
                            "fallback_reason": "",
                            "contract_fingerprint": "risk-fingerprint",
                        },
                    }
                },
                is_paper=True,
                was_executed=True,
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="EDGE/USDT",
                        side="buy",
                        order_type="market",
                        quantity=2.0,
                        price=100.0,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="entry-edge-1",
                        filled_at=datetime(2026, 7, 15, tzinfo=UTC),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="EDGE/USDT",
                        side="long",
                        quantity=2.0,
                        entry_price=100.0,
                        current_price=103.0,
                        leverage=2.0,
                        unrealized_pnl=6.0,
                        is_open=True,
                        entry_exchange_order_id="entry-edge-1",
                        current_management_contract={
                            "contract_version": "2026-07-15.current-position-management.v1",
                            "management_eligible": True,
                            "can_expand_position": False,
                            "can_increase_leverage": False,
                            "blockers": [],
                        },
                        created_at=datetime(2026, 7, 15, tzinfo=UTC),
                    ),
                ]
            )

        payload = await dashboard.get_positions(mode="paper", open_only=True)

        class SplitFakeExecutor:
            async def get_position_protection_orders(self):
                first = dict(protection_order, algo_id="oco-edge-a", contracts="0.75")
                second = dict(protection_order, algo_id="oco-edge-b", contracts="1.25")
                return [first, second]

            async def get_open_orders_strict(self):
                return []

        monkeypatch.setattr(
            dashboard,
            "_dashboard_okx_executor_for_mode",
            lambda _mode: SplitFakeExecutor(),
        )
        split_rows = [{"symbol": "EDGE/USDT", "side": "long"}]
        split_inventory = await dashboard._dashboard_open_position_protection_evidence(
            split_rows,
            mode="paper",
        )

        async def unavailable_risk_evidence(*_args, **_kwargs):
            raise RuntimeError("risk evidence store unavailable")

        async def unavailable_protection_evidence(*_args, **_kwargs):
            raise RuntimeError("OCO inventory unavailable")

        monkeypatch.setattr(
            dashboard,
            "_dashboard_open_position_risk_evidence",
            unavailable_risk_evidence,
        )
        monkeypatch.setattr(
            dashboard,
            "_dashboard_open_position_protection_evidence",
            unavailable_protection_evidence,
        )
        unavailable_payload = await dashboard.get_positions(mode="paper", open_only=True)
    finally:
        await close_db()

    assert payload["count"] == 1
    assert payload["protection_inventory"]["available"] is True
    assert payload["protection_inventory"]["missing_keys"] == []
    row = payload["positions"][0]
    assert row["current_management_contract"]["management_eligible"] is True
    assert row["risk_contract"]["available"] is True
    risk = row["risk_contract"]["contracts"][0]
    assert risk["risk_budget_usdt"] == 4.0
    assert risk["planned_stressed_loss_usdt"] == 3.2
    assert risk["policy_provenance"]["contract_fingerprint"] == "risk-fingerprint"
    assert row["protection_contract"]["available"] is True
    assert row["protection_contract"]["unique"] is True
    assert row["protection_contract"]["orders"][0]["algo_id"] == "oco-edge-1"
    assert row["protection_contract"]["blockers"] == []
    assert split_inventory["split_coverage_keys"] == [["EDGE/USDT", "long"]]
    assert split_rows[0]["protection_contract"]["unique"] is False
    assert split_rows[0]["protection_contract"]["split_coverage"] is True
    assert split_rows[0]["protection_contract"]["coverage_state"] == "split_exact"
    assert split_rows[0]["protection_contract"]["blockers"] == []
    assert unavailable_payload["count"] == 1
    unavailable_row = unavailable_payload["positions"][0]
    assert unavailable_row["risk_contract"]["available"] is False
    assert unavailable_row["risk_contract"]["blockers"] == [
        "position_risk_evidence_unavailable:risk evidence store unavailable"
    ]
    assert unavailable_row["protection_contract"]["available"] is False
    assert unavailable_payload["protection_inventory"]["blockers"] == [
        "okx_protection_evidence_unavailable:OCO inventory unavailable"
    ]


def test_phase11_dashboard_exposes_profit_evidence_without_zero_fallbacks() -> None:
    for token in (
        "mlLocalEvidenceHtml",
        "数据指纹",
        "隔离与降权原因",
        "单币种影响重算",
        "Artifact manifest",
        "晋升与激活证据",
        "mlPredictionEconomicsHtml",
        "原始期望",
        "目标期望",
        "收益下界",
        "不确定性",
        "尾损概率",
        "尾损尺度",
        "往返手续费",
        "权威滑点尾部增量",
        "openPositionEvidenceModal",
        "独立风险预算",
        "计划压力损失",
        "目标 notional",
        "最终 notional",
        "OKX algo ID",
        "全账户保护告警",
        "无效保护单",
        "修复阻断",
        "权威成交结果优先",
        "只作反事实对照，不能覆盖真实盈亏",
    ):
        assert token in SCRIPT

    assert 'status["phase3_new_shadow_sample_count"] = None' in (
        ROOT / "web_dashboard/api/dashboard.py"
    ).read_text(encoding="utf-8")
    assert "不能把缺失显示为 0" in SCRIPT
    assert "readinessDistributionAvailable" in SCRIPT
    assert "dirtySampleRatioLabel" in SCRIPT
    assert "splitEvidenceAvailable ? 'good' : 'warn'" in SCRIPT
    assert "economics.available === true" in SCRIPT
    assert "const pnl = authoritative ? authoritativePnl : fallbackPnl;" in SCRIPT
    assert "authoritativePnl ?? fallbackPnl" not in SCRIPT
    assert "? mlOptionalNumber(authoritative.realized_pnl) : null;" in SCRIPT
    assert "positionProtectionInventoryWarnings(inventory)" in SCRIPT
    assert "精确分片覆盖" in SCRIPT
    assert "multiple_active_okx_protection_orders" not in SCRIPT
    open_positions_block = SCRIPT[
        SCRIPT.index("function renderOpenPositionsTable") : SCRIPT.index(
            "function isOfficialClosedPositionSettlement"
        )
    ]
    assert "positions.map((p, positionIndex) =>" in open_positions_block
    assert 'data-position-index="${positionIndex}"' in open_positions_block
    assert 'id="positions-protection-status"' in HTML
    assert 'id="trade-reflection-authority"' in HTML
    assert "风险 / OCO / 操作" in HTML


def test_phase11_profit_evidence_layout_is_responsive_and_wrap_safe() -> None:
    for selector in (
        ".ml-evidence-grid",
        ".ml-prediction-contract",
        ".position-evidence-grid",
        ".positions-protection-status",
        ".trade-reflection-authority",
        ".trade-outcome-cell",
    ):
        assert selector in STYLE
    assert "overflow-wrap: anywhere" in STYLE
    mobile = STYLE[STYLE.rindex("@media (max-width: 800px)") :]
    assert ".ml-evidence-grid" in mobile
    assert ".position-evidence-grid" in mobile
    assert ".trade-reflection-authority" in mobile


def test_strategy_scheduler_page_separates_production_prior_matches_and_rejections() -> None:
    view = (ROOT / "web_dashboard/static/js/strategy_learning_view.js").read_text(
        encoding="utf-8"
    )
    for token in (
        "所有币种当前共同执行规则",
        "历史收益先验状态 · 不能授权开仓",
        "最近实际匹配记录",
        "rejection_reasons",
        "排名不等于使用",
    ):
        assert token in view
    assert "币种方向候选" not in view


def test_undefined_profit_factor_and_auc_are_not_rendered_as_zero() -> None:
    assert "function profitFactorLabel" in SCRIPT
    assert "function metricNumberLabel" in SCRIPT
    assert "Number(metrics.top_long_profit_factor || 0)" not in SCRIPT
    assert "Number(metrics.top_short_profit_factor || 0)" not in SCRIPT
    assert "Number(row.profit_factor || 0)" not in SCRIPT
    assert "Number(summary.profit_factor || 0)" not in SCRIPT
    assert "Number(metrics.long_auc || 0)" not in SCRIPT
    assert "Number(metrics.short_auc || 0)" not in SCRIPT
    assert "profitFactorLabel(summary.profit_factor)" in SCRIPT
