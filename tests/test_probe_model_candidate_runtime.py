from __future__ import annotations

import pytest

from scripts.probe_model_candidate_runtime import _percentile_95, _valid_contract


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"status":"ok"}', True),
        ('```json\n{"status":"ok"}\n```', True),
        ('{"status":"wrong"}', False),
        ('thinking... {"status":"ok"}', False),
        ("", False),
    ],
)
def test_runtime_probe_requires_exact_json_contract(content: str, expected: bool) -> None:
    assert _valid_contract(content) is expected


def test_runtime_probe_p95_uses_nearest_rank() -> None:
    assert _percentile_95([float(value) for value in range(1, 21)]) == 19.0


def test_runtime_probe_rejects_empty_latency_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        _percentile_95([])
