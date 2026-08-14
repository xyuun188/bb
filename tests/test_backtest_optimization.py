import pandas as pd

from backtest.optimization import (
    ObjectiveConfig,
    ParameterSpace,
    ParameterSpec,
    build_walk_forward_windows,
    parameter_stability_report,
    score_metrics,
    walk_forward_search,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    close = [100.0 + index * 0.1 for index in range(rows)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000.0] * rows,
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
    )


def _space() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterSpec("threshold", "float", minimum=0.1, maximum=0.2, step=0.1),
            ParameterSpec("direction", "choice", values=("long", "short")),
        )
    )


def test_parameter_space_is_deterministic_and_auditable() -> None:
    space = _space()
    first = space.generate(limit=10, random_seed=7)
    second = space.generate(limit=10, random_seed=7)
    assert first == second
    assert len(first) == 4
    assert all(item["candidate_id"].startswith("candidate_") for item in first)
    assert space.fingerprint
    assert len(space.neighbors({"threshold": 0.1, "direction": "long"})) == 2


def test_score_uses_fee_after_risk_objective_and_tail_loss_penalty() -> None:
    score = score_metrics(
        {
            "net_profit": 100,
            "profit_factor": 1.5,
            "max_drawdown_pct": 5,
            "worst_trade_pct": -4,
            "total_trades": 20,
        },
        config=ObjectiveConfig(min_trades=10),
    )
    assert score["eligible"] is True
    assert score["tail_loss_pct"] == 4
    blocked = score_metrics({"net_profit": -1, "total_trades": 1}, config=ObjectiveConfig())
    assert blocked["eligible"] is False
    assert "fee_after_net_profit_not_positive" in blocked["blockers"]


def test_walk_forward_search_records_windows_and_resumes() -> None:
    windows = build_walk_forward_windows(
        80,
        train_rows=30,
        validation_rows=15,
        oos_rows=15,
        step_rows=15,
    )
    calls: list[str] = []

    def evaluator(values: dict, frame: pd.DataFrame, role: str) -> dict:
        calls.append(role)
        base = 10.0 if values["direction"] == "long" else 5.0
        return {
            "net_profit": base + values["threshold"],
            "profit_factor": 1.2,
            "max_drawdown_pct": 2.0,
            "total_trades": 20,
        }

    checkpoints: list[dict] = []
    report = walk_forward_search(
        _frame(),
        parameter_space=_space(),
        windows=windows,
        evaluator=evaluator,
        random_seed=3,
        candidate_limit=4,
        top_k=2,
        objective=ObjectiveConfig(min_trades=10, min_oos_folds=2),
        checkpoint_writer=lambda state: checkpoints.append(state),
    )
    assert report["status"] == "complete"
    assert report["fold_count"] == len(windows)
    assert report["oos_ranking"]
    assert report["oos_ranking"][0]["parameter_stability_passed"] is True
    assert checkpoints
    assert all(fold["status"] == "complete" for fold in report["folds"])

    calls_before_resume = len(calls)
    resumed = walk_forward_search(
        _frame(),
        parameter_space=_space(),
        windows=windows,
        evaluator=evaluator,
        random_seed=3,
        candidate_limit=4,
        top_k=2,
        objective=ObjectiveConfig(min_trades=10, min_oos_folds=2),
        resume_state=report["resume_state"],
    )
    assert len(calls) == calls_before_resume
    assert resumed["optimization_id"] == report["optimization_id"]


def test_parameter_stability_rejects_cliff() -> None:
    stable = parameter_stability_report(100.0, {"near": 80.0})
    cliff = parameter_stability_report(100.0, {"bad": 20.0})
    assert stable["status"] == "pass"
    assert cliff["status"] == "fail"
