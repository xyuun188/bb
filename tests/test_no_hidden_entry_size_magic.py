from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_DIRS = (
    "ai_brain",
    "services",
    "executor",
    "risk_manager",
)

FORBIDDEN_PATTERNS = (
    r"ML_SOFT_CAUTION_MAX_ENTRY_SIZE\s*=\s*0\.025",
    r"PROFIT_FIRST_PROBE_SIZE\s*=\s*0\.025",
    r"QUANT_VALIDATION_PROBE_SIZE\s*=\s*0\.04",
    r"BALANCED_PROBE_MAX_POSITION_SIZE_PCT\s*=\s*0\.018",
    r"size\s*=\s*min\(size,\s*0\.018\)",
    r"crowded_cap\s*=\s*0\.025",
    r"size_cap\s*=\s*0\.02",
    r"size_cap\s*=\s*0\.025",
    r"size_cap\s*=\s*0\.015",
    r"max_probe_size_pct\"\s*:\s*0\.018",
    r"setdefault\(\"max_probe_size_pct\",\s*0\.018\)",
    r"probe_size\s*=\s*0\.060",
    r"stop_loss_pct\s*=\s*0\.012",
)


def test_entry_size_caps_are_not_hidden_magic_numbers() -> None:
    offenders: list[str] = []
    compiled = [re.compile(pattern) for pattern in FORBIDDEN_PATTERNS]
    for dirname in PRODUCTION_DIRS:
        for path in (PROJECT_ROOT / dirname).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                if any(pattern.search(line) for pattern in compiled):
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{line_number}: {line.strip()}"
                    )

    assert offenders == []
