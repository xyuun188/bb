from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCAN_TARGETS = (
    ("ai_brain", "*.py"),
    ("config", "*.py"),
    ("core", "*.py"),
    ("db", "*.py"),
    ("models", "*.py"),
    ("services", "*.py"),
    ("web_dashboard/api", "*.py"),
    ("web_dashboard", "app.py"),
    ("web_dashboard/static/js", "*.js"),
)

ALLOWED_FILES = {
    Path("web_dashboard/api/text_sanitize.py"),
    Path("tests/test_batch_expert_json_stability.py"),
    Path("tests/test_expert_memory_cleanup.py"),
    Path("tests/test_expert_memory_service.py"),
    Path("tests/test_strategy_learning.py"),
}

MOJIBAKE_MARKERS = (
    "锟",
    "锛",
    "銆",
    "閫",
    "閸",
    "閹",
    "鈧",
    "鈥",
    "鏈",
    "鏃",
    "璇",
    "鍒",
    "鍙",
    "瑙",
    "鎴",
    "鏁",
    "绛",
    "浣犺",
    "浜忔崯",
    "鐩",
    "绠€",
    "",
    "",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
)


def test_runtime_source_does_not_contain_mojibake_literals() -> None:
    offenders: list[str] = []
    for dirname, pattern in SCAN_TARGETS:
        for path in (PROJECT_ROOT / dirname).rglob(pattern):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = sorted({marker for marker in MOJIBAKE_MARKERS if marker in text})
            if hits:
                offenders.append(f"{rel}: {', '.join(hits)}")

    assert offenders == []
