from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


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
    ("web_dashboard/static/css", "*.css"),
    ("web_dashboard/static", "*.html"),
    ("scripts", "*.py"),
)

ALLOWED_FILES = {
    Path("tests/test_batch_expert_json_stability.py"),
    Path("tests/test_expert_memory_cleanup.py"),
    Path("tests/test_expert_memory_service.py"),
    Path("tests/test_strategy_learning.py"),
}

MOJIBAKE_MARKERS = (
    _u("\\u951f"),
    _u("\\u951b"),
    _u("\\u9286"),
    _u("\\u95ab"),
    _u("\\u95b8"),
    _u("\\u95b9"),
    _u("\\u9227"),
    _u("\\u9225"),
    _u("\\u93c8"),
    _u("\\u93c3"),
    _u("\\u7487"),
    _u("\\u9352"),
    _u("\\u9359"),
    _u("\\u7459"),
    _u("\\u93b4"),
    _u("\\u93c1"),
    _u("\\u7edb"),
    _u("\\u6d63\\u72ba"),
    _u("\\u6d5c\\u5fd4\\u5d2f"),
    _u("\\u9429"),
    _u("\\u7ee0\\u20ac"),
    _u("\\ue750"),
    _u("\\ue6e6"),
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
