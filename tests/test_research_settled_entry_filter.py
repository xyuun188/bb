import json
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from scripts import research_settled_entry_filter as research

T = datetime(2026, 11, 12, tzinfo=UTC)


def outcome():
    return {
        "execution_mode": "paper", "close_status": "full", "trade_fact_trusted": True,
        "outcome_complete": True, "strategy_entry_supervision_eligible": True,
        "realized_net_pnl_components": {"formula_consistent": True, "reported_realized_net_pnl_usdt": -5},
        "net_return_after_all_cost_pct": -0.7, "decision_id": 8, "entry_decision_ids": [8],
        "lifecycle_key": "abc", "side": "short", "decision_timestamp": T.isoformat(),
        "label_timestamp": (T + timedelta(hours=1)).isoformat(),
        "features": dict.fromkeys(research.FEATURES, 2.),
    }


@pytest.mark.parametrize("changes", [
    {"execution_mode": "live"}, {"entry_decision_ids": [7, 8]}, {"close_status": "partial"},
    {"trade_fact_trusted": False}, {"strategy_entry_supervision_eligible": False},
    {"outcome_complete": False}, {"side": None}, {"label_timestamp": T.replace(tzinfo=None).isoformat()},
])
def test_filters_invalid_paths(changes):
    assert research.sample(outcome() | changes, T)[0] is None


def test_sealed_features_and_signed_actual_return_only(monkeypatch):
    row = outcome()
    assert research.sample(row, T)[1] == "entry_features_not_sealed"
    monkeypatch.setattr(research, "feature_availability_reasons", lambda *a, **k: [])
    late = T + timedelta(days=2)
    item, error = research.sample(row | {"counterfactual_evidence": 100}, late)
    assert not error
    assert item["known"] == late.isoformat() and item["holding_minutes"] == 60
    assert item["x"] == (-1, -2, -2, -2, 2, 2, 2)
    assert (item["net"], item["pnl"]) == (-0.7, -5)
    assert research.sample(row, None)[0] is None


def test_fitting_purges_late_labels_and_rejects_overlap(monkeypatch):
    monkeypatch.setattr(research, "feature_availability_reasons", lambda *a, **k: [])
    split = T + timedelta(days=1)
    items = []
    for i in range(34):
        side = "short" if i % 2 else "long"
        r = outcome() | {"decision_id": i + 1, "entry_decision_ids": [i + 1], "side": side,
                         "lifecycle_key": str(i), "net_return_after_all_cost_pct": 1 if i % 2 else -1}
        items.append(research.sample(r, T if i < 32 else split)[0])
    model = research.fit(items, T, split)
    assert model["train_decisions"] == list(range(1, 33))
    assert model["excluded_after_cutoff"] == 2
    with pytest.raises(ValueError, match="overlap"):
        research.evaluate(model, items)
    result = research.evaluate(model, items[-2:])
    assert result["kept"]["sample_count"] == 1
    assert result["kept"]["realized_net_pnl_usdt"] == -5
    assert result["not_selected"]["sample_count"] == 1
    assert result["promotable"] is False
    assert research.evaluate(model, [])["kept"]["realized_net_pnl_usdt"] is None


@pytest.mark.asyncio
async def test_evaluation_only_uses_settlements_known_before_end(tmp_path, monkeypatch):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    split = start + timedelta(days=1)
    end = split + timedelta(days=1)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({
        "version": research.VERSION, "features": research.INPUTS, "start": start.isoformat(),
        "split": split.isoformat(), "frozen_at": split.isoformat(), "available_training_count": 32,
        "training_exclusions": {}, "excluded_after_cutoff": 0,
    }))

    async def load(lower, upper):
        assert (lower, upper) == (split, end)
        return [{"known": (end + timedelta(seconds=offset)).isoformat(), "offset": offset}
                for offset in (-1, 0, 1)], {}

    monkeypatch.setattr(research, "load", load)
    monkeypatch.setattr(research, "evaluate", lambda model, rows: {"offsets": [r["offset"] for r in rows]})
    report = await research.run(Namespace(start=start, split=split, end=end, forward=True, artifact=path))
    assert report["offsets"] == [-1]
    assert report["evaluation_exclusions"]["mirror_at_or_after_cutoff"] == 2
