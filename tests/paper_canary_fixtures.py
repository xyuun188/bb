from __future__ import annotations

from typing import Any

from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PAPER_BOOTSTRAP_SIZING_VERSION,
)


def complete_paper_canary_raw() -> dict[str, Any]:
    sampling_stratum = {
        "symbol": "BTC/USDT",
        "side": "long",
        "volatility_bucket": "medium",
        "market_regime": "trending",
        "key": "BTC/USDT|long|medium|trending",
    }
    canary_provenance = {
        "source": "governed_paper_model_distribution_for_normal_strategy_trading",
        "observation_window": "current_model_artifact_and_pre_order_market_snapshot",
        "sample_count": 1200,
        "generated_at": "2026-07-17T08:00:00+00:00",
        "strategy_version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "fallback_reason": "",
    }
    sizing_provenance = {
        "source": "paper_normal_strategy_independent_risk_budget",
        "observation_window": "current_okx_demo_account_market_and_normal_risk_guard",
        "sample_count": 1200,
        "generated_at": "2026-07-17T08:01:00+00:00",
        "strategy_version": PAPER_BOOTSTRAP_SIZING_VERSION,
        "fallback_reason": "",
        "contract_fingerprint": "canary-contract-fingerprint",
    }
    return {
        "pre_order_execution_facts": {
            "production_eligible": True,
            "input_fingerprint": "paper-canary-pre-order-fingerprint",
            "inst_id": "BTC-USDT-SWAP",
            "contract_spec": {
                "ctVal": "0.01",
                "ctMult": "1",
                "source": "okx_public_instruments",
            },
        },
        "paper_bootstrap_canary": {
            "version": PAPER_BOOTSTRAP_CANARY_VERSION,
            "authorized": True,
            "requested": True,
            "execution_scope": "paper_only",
            "production_permission": False,
            "purpose": "execute_normal_paper_strategy_and_learn_after_settlement",
            "trade_kind": "normal_strategy_trade",
            "position_exit_policy": "dynamic_strategy_risk_and_position_review",
            "continuous_training_after_settlement": True,
            "selected_side": "long",
            "selected_observation": {
                "side": "long",
                "raw_expected_return_pct": 0.35,
                "objective_expected_return_pct": 0.15,
                "lower_quantile_return_pct": 0.05,
                "dispersion_pct": 0.1,
                "current_execution_cost_pct": 0.08,
                "observed_net_return_pct": 0.27,
                "lower_quantile_net_return_pct": -0.03,
                "horizon_minutes": 10,
                "distribution_member_count": 128,
                "source_authority": "extra_trees_empirical_distribution",
                "sampling_stratum": sampling_stratum,
            },
            "sampling_stratum": sampling_stratum,
            "direction_score_gap": 0.23,
            "confidence": 0.6969697,
            "artifact_version": "candidate-v1",
            "artifact_lifecycle": "canary",
            "source_sample_count": 1200,
            "runtime_authorized": True,
            "runtime_guard": {
                "blocking_reasons": [],
                "open_position_count": 0,
                "max_open_positions": 1,
                "daily_entry_count": 0,
                "max_daily_entries": 4,
            },
            "policy_provenance": canary_provenance,
        },
        "opportunity_score": {
            "score": None,
            "production_eligible": False,
        },
        "profit_risk_sizing": {
            "contract_version": PAPER_BOOTSTRAP_SIZING_VERSION,
            "contract_lifecycle": "paper_bootstrap_canary",
            "execution_scope": "paper_only",
            "production_permission": False,
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 500.0,
            "risk_budget_usdt": 0.5,
            "single_trade_risk_budget_usdt": 0.5,
            "portfolio_risk_budget_usdt": 1.0,
            "planned_stressed_loss_usdt": 0.5,
            "stressed_loss_fraction": 0.01,
            "target_notional_usdt": 50.0,
            "final_notional_usdt": 50.0,
            "final_margin_usdt": 50.0,
            "position_size_pct": 0.1,
            "portfolio_risk_snapshot": {
                "scope": "paper_account_positions",
                "current_stressed_loss_usdt": 0.0,
            },
            "entry_instrument_availability": {"available": True},
            "leverage_tier_selection": {"production_eligible": True},
            "policy_provenance": sizing_provenance,
        },
    }


def bounded_legacy_fill_drift_raw(
    *,
    excess_fraction: float = 0.001,
) -> dict[str, Any]:
    raw = complete_paper_canary_raw()
    sizing = raw["profit_risk_sizing"]
    target = float(sizing["target_notional_usdt"])
    settled = target * (1.0 + excess_fraction)
    drift_reasons = [
        "execution_notional_exceeds_authoritative_target",
        "execution_stressed_loss_exceeds_risk_budget",
    ]
    raw["paper_bootstrap_canary"]["selected_observation"][
        "current_execution_cost_pct"
    ] = 0.2
    raw["opportunity_score"]["execution_cost"] = {"total_pct": 0.1}
    sizing.update(
        {
            "production_eligible": False,
            "reason": ",".join(drift_reasons),
            "final_notional_usdt": settled,
            "final_margin_usdt": settled,
            "planned_stressed_loss_usdt": settled
            * float(sizing["stressed_loss_fraction"]),
            "position_size_pct": 0.0,
            "execution_reconciliations": [
                {
                    "source": "okx_pre_submit_order_shape",
                    "final_notional_usdt": target - 0.01,
                    "eligible": True,
                    "reasons": [],
                },
                {
                    "source": "okx_confirmed_entry_fill",
                    "final_notional_usdt": settled,
                    "eligible": False,
                    "reasons": drift_reasons,
                },
            ],
        }
    )
    sizing["policy_provenance"]["fallback_reason"] = ",".join(drift_reasons)
    return raw
