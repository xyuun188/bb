import json

import pytest

from scripts.research_abstaining_direction import FEATURES, fit, observation, predict


@pytest.mark.parametrize("trend,expected", [(1, (1, -2)), (-1, (-2, 1))])
def test_features_are_decision_only_and_targets_follow_trend(trend, expected):
    snap = dict.fromkeys(FEATURES, 1)
    snap.update(change_24h_pct=trend, future_outcome=1000)
    row = dict(training_eligible=True, feature_snapshot=snap, long_net_return_after_all_cost_pct=1,
               short_net_return_after_all_cost_pct=-2, source_id=1, decision_id=10,
               symbol="X", decision_timestamp="t0", label_timestamp="t1")
    observed = observation(row)
    assert observed["net"] == expected and 1000 not in observed["x"]
    assert observation({**row, "training_eligible": False}) is None
    assert observation({**row, "feature_snapshot": {}}) is None


def test_frozen_tree_can_continue_reverse_and_abstain_without_exam_labels():
    samples = [{"x": (value,) * len(FEATURES), "net": returns}
               for value, returns in [(1, (1, -1)), (2, (-1, 1)), (3, (-1, -1))]
               for _ in range(40)]
    model = json.loads(json.dumps(fit(samples)))
    for trend, sides in [(0, ("long", "short", "hold")), (1, ("short", "long", "hold"))]:
        for value, side in enumerate(sides, 1):
            assert predict(model, {"x": (value,) * len(FEATURES), "trend": trend}) == side
