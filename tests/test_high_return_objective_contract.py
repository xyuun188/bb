from __future__ import annotations

import re
from pathlib import Path

import pytest

from services.return_objective import profit_factor

ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("relative_path", "patterns"),
    [
        (
            "services/ml_readiness.py",
            (
                r"auc_below_threshold",
                r"pr_auc_below_threshold",
                r"accuracy_below_threshold",
                r"top_win_not_above",
            ),
        ),
        (
            "services/ml_signal_service.py",
            (
                r"_expected_return_from_win_probability",
                r"expected_return_calibration",
                r"MIN_PROFIT_SIGNAL_WIN_RATE",
                r"winrate_model",
            ),
        ),
        (
            "services/specialist_shadow_evaluation.py",
            (r'get\("net_return_after_cost_pct",\s*[^\n]*return_after_cost_pct',),
        ),
        (
            "services/training_data_quality.py",
            (r'get\("net_return_after_cost_pct",\s*[^\n]*return_after_cost_pct',),
        ),
        (
            "db/repositories/memory_repo.py",
            (r'get\("net_return_after_cost_pct",\s*[^\n]*pnl_pct',),
        ),
        (
            "services/trading_params.py",
            (
                r"min_profit_signal_win_rate",
                r"influence_min_auc",
                r"influence_min_pr_auc",
                r"influence_min_accuracy",
                r"ml_min_support_win_rate",
                r"ml_strong_support_win_rate",
                r"ml_low_win_confidence_bonus",
                r"ml_profit_first_low_win_rate_size_multiplier",
            ),
        ),
        ("services/entry_opportunity_scoring.py", (r"win_rate\s*\*",)),
        ("services/expert_memory_service.py", (r"win_component",)),
        (
            "services/expert_memory_service.py",
            (r'net_return_after_cost_pct"\s*:\s*pnl_pct\s*[,}]',),
        ),
        ("services/model_contribution_performance.py", (r"win_edge",)),
        (
            "ai_brain/ensemble_coordinator.py",
            (
                r"blocked_low_ml_win_rate",
                r"passed_high_confidence_low_win_rate",
                r"ML_MIN_SUPPORT_WIN_RATE",
                r"ML_STRONG_SUPPORT_WIN_RATE",
            ),
        ),
        (
            "services/entry_direction_competition.py",
            (r"win_rate\s*\*", r"1\.0\s*-\s*win_rate"),
        ),
        ("services/dynamic_position_capacity.py", (r"if\s+recent_win_rate\s*[<>]",)),
        (
            "services/entry_strategy_mode.py",
            (r"win_rate\s*[<>]=", r"md_win_rate\s*[<>]="),
        ),
        (
            "services/competition_service.py",
            (r"score\s*=\s*\([^)]*win_rate", r"decision_accuracy\s*\*\s*0\."),
        ),
    ],
)
def test_win_rate_does_not_flow_into_production_decisions(
    relative_path: str,
    patterns: tuple[str, ...],
) -> None:
    source = _source(relative_path)
    for pattern in patterns:
        assert re.search(pattern, source, re.DOTALL) is None, (relative_path, pattern)


def test_local_quant_training_has_no_hindsight_best_direction_target() -> None:
    source = _source("scripts/deploy_local_ai_tools_service.py")

    assert 'max(r["long_return"], r["short_return"], key=abs)' not in source
    assert (
        'max(net_return_pct(f(sample, "long_return_pct")), '
        'net_return_pct(f(sample, "short_return_pct")))'
        not in source
    )
    assert '"long_model": long_model' in source
    assert '"short_model": short_model' in source


@pytest.mark.parametrize(
    "relative_path",
    (
        "services/entry_opportunity_scoring.py",
        "services/entry_profit_risk_sizing.py",
        "services/return_objective.py",
    ),
)
def test_live_entry_pipeline_cannot_read_retired_return_target(
    relative_path: str,
) -> None:
    source = _source(relative_path)

    assert '"net_return_after_cost_pct"' not in source
    assert '"net_return_after_all_cost_pct"' in source


def test_finquant_uses_sft_then_trl_dpo_return_preferences() -> None:
    source = _source("scripts/finquant_expert_lora_training.py")

    assert "from trl import DPOConfig, DPOTrainer" in source
    assert "low_win_high_payoff_vs_high_win_negative_expectancy" in source
    assert '"sft_format_domain", "trl_dpo_return_preference"' in source
    assert "format_and_language_fit_only_not_profit_evidence" in source


def test_low_win_high_payoff_beats_high_win_negative_expectancy() -> None:
    low_win_returns = [4.0] * 35 + [-1.0] * 65
    high_win_returns = [0.1] * 80 + [-2.0] * 20

    assert sum(low_win_returns) / len(low_win_returns) == pytest.approx(0.75)
    assert profit_factor(low_win_returns) == pytest.approx(35 * 4.0 / 65)
    assert sum(high_win_returns) / len(high_win_returns) == pytest.approx(-0.32)
    assert profit_factor(high_win_returns) == pytest.approx(0.2)


def test_profit_factor_is_undefined_without_a_loss_denominator() -> None:
    assert profit_factor([0.8, 0.3, 1.1]) is None
    assert profit_factor([0.0, 0.0]) is None
