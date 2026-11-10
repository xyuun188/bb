"""Calendar-block uncertainty for correlated development opportunities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_VERSION = "2026-11-10.calendar-block-return-uncertainty.v1"
MIN_CALENDAR_BLOCKS = 16
BLOCK_LENGTH_DAYS = 2
BOOTSTRAP_REPLICATIONS = 2000
RANDOM_SEED = 42


def blocked_return_evidence(frame: pd.DataFrame, sides: np.ndarray) -> dict[str, Any]:
    """Resample adjacent UTC days jointly across symbols, including hold days.

    This is a development filter. Calendar blocking reduces pseudoreplication;
    it neither makes data independent nor supplies execution or exam evidence.
    """

    if len(frame) != len(sides):
        raise ValueError("return selections must match frame length")
    if not set(np.asarray(sides)).issubset({"hold", "long", "short"}):
        raise ValueError("invalid selected return direction")
    report: dict[str, Any] = {
        "version": POLICY_VERSION,
        "method": "moving_two_day_calendar_blocks_all_symbols_jointly",
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "random_seed": RANDOM_SEED,
        "minimum_calendar_days": MIN_CALENDAR_BLOCKS,
        "calendar_day_count": 0,
        "selected_count": 0,
        "return_lcb_pct": None,
        "ready": False,
        "reason": "insufficient_calendar_coverage",
        "independent_evidence_available": False,
        "executed_portfolio_pnl": False,
    }
    if frame.empty:
        return report
    times = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("missing decision clock for calendar return evidence")
    selected = np.isin(sides, ("long", "short"))
    returns = np.zeros(len(frame), dtype=float)
    for side in ("long", "short"):
        mask = np.asarray(sides) == side
        returns[mask] = (
            frame.loc[mask, f"{side}_return_pct"].to_numpy(dtype=float)
            - frame.loc[mask, f"{side}_execution_cost_pct"].to_numpy(dtype=float)
        )
    if not np.isfinite(returns).all():
        raise ValueError("non-finite calendar return")
    days = times.dt.floor("D")
    calendar = pd.date_range(days.min(), days.max(), freq="D")
    daily = pd.DataFrame({
        "day": days.to_numpy(), "net": returns, "selected": selected.astype(int),
    }).groupby("day")[["net", "selected"]].sum().reindex(calendar, fill_value=0)
    report["calendar_day_count"] = len(daily)
    report["selected_count"] = int(selected.sum())
    report["selected_calendar_day_count"] = int(daily["selected"].gt(0).sum())
    if (
        len(daily) < MIN_CALENDAR_BLOCKS
        or report["selected_calendar_day_count"] < MIN_CALENDAR_BLOCKS
    ):
        return report
    rng = np.random.default_rng(RANDOM_SEED)
    starts = rng.integers(
        0, len(daily) - BLOCK_LENGTH_DAYS + 1,
        size=(BOOTSTRAP_REPLICATIONS, int(np.ceil(len(daily) / BLOCK_LENGTH_DAYS))),
    )
    indices = (
        starts[:, :, None] + np.arange(BLOCK_LENGTH_DAYS)[None, None, :]
    ).reshape(BOOTSTRAP_REPLICATIONS, -1)[:, :len(daily)]
    totals = daily["net"].to_numpy()[indices].sum(axis=1)
    counts = daily["selected"].to_numpy()[indices].sum(axis=1)
    if (counts == 0).any():
        report["reason"] = "resampled_blocks_without_selections"
        return report
    report.update({
        "return_lcb_pct": round(float(np.quantile(totals / counts, 0.05)), 12),
        "ready": True,
        "reason": "development_block_uncertainty_computed",
    })
    return report
