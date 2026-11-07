"""Fixed low-turnover hypotheses on development data, without execution access."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from core.symbols import normalize_trading_symbol
from services.ml_development_baselines import _development_input_sha256, _evidence
from services.ml_horizon_development import _folds

POLICY_VERSION = "2026-11-07.fixed-liquidity-regime-development.v1"
HORIZONS = (60, 240)
MIN_SELECTED_COUNT = 16
MAX_CONCURRENT_POSITIONS = 3
LIQUIDITY_FILTERS = {
    "liquid": {"minimum_notional_usdt": 10_000_000, "maximum_spread_pct": 0.05},
    "tight": {"minimum_notional_usdt": 100_000_000, "maximum_spread_pct": 0.02},
}
CANDIDATES = tuple(
    {
        "id": f"{signal}-{liquidity}-{horizon}m",
        "signal": signal,
        "liquidity": liquidity,
        "horizon_minutes": horizon,
    }
    for horizon, signal, liquidity in product(
        HORIZONS, ("trend", "reversion"), LIQUIDITY_FILTERS,
    )
)


def protocol() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "candidates": list(CANDIDATES),
        "liquidity_filters": LIQUIDITY_FILTERS,
        "signal_rules": {
            "trend": (
                "regime_trending=1; long when returns_20>0 and price_vs_sma50>0; "
                "short when both are negative"
            ),
            "reversion": (
                "regime_ranging=1; long when rsi_14<=40 and price_vs_sma20<0; "
                "short when rsi_14>=60 and price_vs_sma20>0"
            ),
            "common": (
                "positive finite spread within registered cap; contemporaneous "
                "notional above floor; entry_activity_volume_ratio>=1; "
                "finite positive volatility_20; 0<rsi_14<100; "
                "no label, realized cost, or future feature enters signal"
            ),
        },
        "position_policy": (
            "at most three concurrent positions, one per canonical symbol; "
            "fixed horizon exit; deterministic decision-time/group/id priority; "
            "capacity carries across validation fold boundaries"
        ),
        "cost_policy": "native fee-after-complete labels, including signed funding",
        "fold_policy": (
            "common 60m/240m decision groups; first half development history, "
            "last half three chronological folds; training labels strictly "
            "before validation; no fitted parameters or outcome-based selection"
        ),
        "acceptance_policy": (
            "each fold and aggregate, both all-opportunity and capacity views: "
            "at least 16 selected, positive fee-after mean and naive LCB, PF>1; "
            "all eight hypotheses reported, no automatic promotion"
        ),
        "minimum_selected_count": MIN_SELECTED_COUNT,
        "maximum_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    }


def candidate_sides(frame: pd.DataFrame, candidate: dict[str, Any]) -> np.ndarray:
    """Select using frozen contemporaneous features only."""

    limits = LIQUIDITY_FILTERS[candidate["liquidity"]]
    required = [
        "spread_pct", "log_notional_24h_usdt", "entry_activity_volume_ratio",
        "volatility_20", "rsi_14", "returns_20", "price_vs_sma50",
        "price_vs_sma20", "regime_trending", "regime_ranging",
    ]
    values = frame[required].astype(float)
    available = np.isfinite(values).all(axis=1)
    eligible = (
        available
        & (frame["spread_pct"] > 0)
        & (frame["spread_pct"] <= limits["maximum_spread_pct"])
        & (frame["log_notional_24h_usdt"] >= np.log10(limits["minimum_notional_usdt"] + 1))
        & (frame["entry_activity_volume_ratio"] >= 1)
        & (frame["volatility_20"] > 0)
        & (frame["rsi_14"] > 0) & (frame["rsi_14"] < 100)
    )
    if candidate["signal"] == "trend":
        eligible &= frame["regime_trending"] == 1
        long = (frame["returns_20"] > 0) & (frame["price_vs_sma50"] > 0)
        short = (frame["returns_20"] < 0) & (frame["price_vs_sma50"] < 0)
    elif candidate["signal"] == "reversion":
        eligible &= frame["regime_ranging"] == 1
        long = (frame["rsi_14"] <= 40) & (frame["price_vs_sma20"] < 0)
        short = (frame["rsi_14"] >= 60) & (frame["price_vs_sma20"] > 0)
    else:
        raise ValueError("unknown registered signal")
    return np.select([eligible & long, eligible & short], ["long", "short"], default="hold")


def capacity_sides(frame: pd.DataFrame, sides: np.ndarray) -> np.ndarray:
    """Reserve fixed-horizon slots without consulting labels or observed exits."""

    decisions = pd.to_datetime(frame["decision_timestamp"], utc=True)
    symbols = frame["symbol"].map(
        lambda value: normalize_trading_symbol("" if pd.isna(value) else value)
    )
    selected = np.full(len(frame), "hold", dtype=object)
    occupied: dict[str, pd.Timestamp] = {}
    order = sorted(range(len(frame)), key=lambda index: (
        decisions.iloc[index], str(frame.iloc[index]["decision_group"]),
        int(frame.iloc[index]["id"]),
    ))
    for index in order:
        if sides[index] not in {"long", "short"} or not symbols.iloc[index]:
            continue
        now = decisions.iloc[index]
        occupied = {symbol: end for symbol, end in occupied.items() if end > now}
        symbol = symbols.iloc[index]
        if symbol in occupied or len(occupied) >= MAX_CONCURRENT_POSITIONS:
            continue
        selected[index] = sides[index]
        occupied[symbol] = now + pd.Timedelta(minutes=int(frame.iloc[index]["horizon_minutes"]))
    return selected


def _views(frame: pd.DataFrame, sides: np.ndarray, executable: np.ndarray) -> dict[str, Any]:
    return {
        "all_opportunities": _evidence(frame, sides),
        "capacity": _evidence(frame, executable),
    }


def _rejections(views: dict[str, Any]) -> list[str]:
    reasons = []
    for name, view in views.items():
        if view["selected_count"] < MIN_SELECTED_COUNT:
            reasons.append(f"{name}_insufficient_selected_count")
        if (view.get("avg_return_pct") or 0) <= 0:
            reasons.append(f"{name}_nonpositive_net_mean")
        if (view.get("return_lcb_pct") or 0) <= 0:
            reasons.append(f"{name}_nonpositive_naive_lcb")
        if (view.get("profit_factor") or 0) <= 1:
            reasons.append(f"{name}_profit_factor_not_above_one")
    return reasons


def build_liquidity_regime_development_report(
    development: pd.DataFrame, *, snapshot: dict[str, Any],
) -> dict[str, Any]:
    if _development_input_sha256(development) != snapshot["development_input_sha256"]:
        raise ValueError("research must use the exact frozen development frame")
    if development.duplicated(["decision_group", "horizon_minutes"]).any():
        raise ValueError("one observation per group and horizon required")
    coverage = development.groupby("decision_group")["horizon_minutes"].agg(set)
    complete = set(coverage[coverage.map(lambda values: set(HORIZONS) <= values)].index)
    common = development[
        development["decision_group"].isin(complete)
        & development["horizon_minutes"].isin(HORIZONS)
    ].copy()
    report: dict[str, Any] = {
        "status": "insufficient_development_data",
        "reason": "insufficient_common_horizon_groups",
        "protocol": protocol(),
        "common_horizon_decision_group_count": len(complete),
        "excluded_incomplete_group_count": len(coverage) - len(complete),
        "common_horizon_row_count": len(common),
        "diagnostic_only": True,
        "exam_labels_evaluated": False,
        "artifact_persisted": False,
        "promotion_authorized": False,
        "orders_submitted": False,
        "runtime_policy_changed": False,
        "metric_limitations": (
            "development counterfactuals, not settled portfolio PnL; concurrent "
            "symbols and sequential trades may be correlated; naive LCB is "
            "a development filter, not independent profitability evidence"
        ),
        "folds": [], "candidates": [], "qualified_candidate_ids": [],
    }
    if common.empty:
        return report
    folds = _folds(common, 3)
    if len(folds) != 3 or any(boundary["training_sample_count"] < MIN_SELECTED_COUNT
                              for _, _, boundary in folds):
        report["reason"] = "insufficient_purged_chronological_folds"
        return report
    report["folds"] = [boundary for _, _, boundary in folds]
    evaluation = pd.concat([validation for _, validation, _ in folds], ignore_index=True)
    report["fixed_baselines_by_horizon"] = {}
    for horizon in HORIZONS:
        frame = evaluation[evaluation["horizon_minutes"] == horizon].reset_index(drop=True)
        report.setdefault("contemporaneous_coverage_by_horizon", {})[str(horizon)] = {
            "opportunity_count": len(frame),
            "trend_regime_count": int((frame["regime_trending"] == 1).sum()),
            "ranging_regime_count": int((frame["regime_ranging"] == 1).sum()),
            "positive_spread_count": int((frame["spread_pct"] > 0).sum()),
            "active_volume_count": int((frame["entry_activity_volume_ratio"] >= 1).sum()),
            "positive_volatility_count": int((frame["volatility_20"] > 0).sum()),
            "median_spread_pct": float(frame["spread_pct"].median()),
            "median_notional_log10_usdt": float(frame["log_notional_24h_usdt"].median()),
        }
        fold_masks = [
            frame["decision_group"].isin(validation["decision_group"]).to_numpy()
            for _, validation, _ in folds
        ]
        baselines = {}
        for side in ("hold", "long", "short"):
            sides = np.full(len(frame), side, dtype=object)
            executable = capacity_sides(frame, sides)
            baselines[side] = _views(frame, sides, executable)
        report["fixed_baselines_by_horizon"][str(horizon)] = baselines
        for candidate in (item for item in CANDIDATES if item["horizon_minutes"] == horizon):
            sides = candidate_sides(frame, candidate)
            executable = capacity_sides(frame, sides)
            aggregate = _views(frame, sides, executable)
            per_fold = []
            rejections = [f"aggregate:{reason}" for reason in _rejections(aggregate)]
            for index, mask in enumerate(fold_masks):
                views = _views(frame.loc[mask], sides[mask], executable[mask])
                reasons = _rejections(views)
                rejections.extend(f"fold_{index}:{reason}" for reason in reasons)
                per_fold.append({"fold_index": index, "evidence": views, "rejection_reasons": reasons})
            passed = not rejections
            report["candidates"].append({
                "policy": candidate,
                "development_qualified": passed,
                "rejection_reasons": rejections,
                "aggregate_evidence": aggregate,
                "folds": per_fold,
            })
            if passed:
                report["qualified_candidate_ids"].append(candidate["id"])
    report["status"] = (
        "development_candidate_qualified" if report["qualified_candidate_ids"] else "rejected"
    )
    report["reason"] = (
        "registered_fee_after_rules_passed"
        if report["qualified_candidate_ids"] else "no_registered_policy_qualified"
    )
    return report
