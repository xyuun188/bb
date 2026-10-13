from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_preserves_low_price_changes_and_missing_values() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard formatting verification")
    script = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    start = script.index("function fmtPrice(")
    end = script.index("\n}", start) + 2
    program = script[start:end] + """
console.log(JSON.stringify([
    0.00000403, 0.00000412, 0.05263, 123.4567, null, undefined, '', 0, NaN
].map(fmtPrice)));
"""
    result = subprocess.run(  # noqa: S603
        [node, "-e", program], check=True, capture_output=True, text=True, timeout=10,
    )
    assert json.loads(result.stdout) == [
        "0.000004030", "0.000004120", "0.05263", "123.4567",
        "-", "-", "-", "-", "-",
    ]
