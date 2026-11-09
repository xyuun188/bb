from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "services" / "daily_briefing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bb_daily_briefing", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocking_reasons_reads_permission_and_selection() -> None:
    module = _load_module()

    reasons = module._blocking_reasons(
        {
            "entry_permission": {
                "blocking_reasons": [
                    "direction_support_expected_net_not_positive",
                    "direction_support_quant_evidence_missing",
                ]
            },
            "paper_trade_selection": {
                "selected": False,
                "selection_reason": "no_direction",
            },
        }
    )

    assert reasons == [
        "direction_support_expected_net_not_positive",
        "direction_support_quant_evidence_missing",
        "paper_trade_selection:no_direction",
    ]


def test_blocking_reasons_is_empty_for_non_mapping_snapshot() -> None:
    module = _load_module()

    assert module._blocking_reasons(None) == []
    assert module._blocking_reasons("not-a-snapshot") == []


def test_render_text_reports_blockers_and_model_state() -> None:
    module = _load_module()
    report = {
        "window_hours": 24,
        "decisions": {
            "in_window": 300,
            "executed": 0,
            "actions": {"hold": 300},
            "top_reasons": [
                ("direction_support_expected_net_not_positive", 300),
                ("paper_trade_selection:no_direction", 300),
            ],
        },
        "shadow": {"completed_in_window": 12},
        "model": {
            "lifecycle": "canary",
            "artifact_version": "20261108T233138005216Z-95ecdd20",
            "paper_model_eligible": True,
            "live_ml_ready": False,
            "shadow_samples": 176522,
            "trade_samples": 107,
            "message": "",
        },
    }

    text = module.render_text(report)

    assert "决策 300 次" in text
    assert "实际开仓 0 次" in text
    assert "direction_support_expected_net_not_positive" in text
    assert "canary" in text
    assert "实盘权限：False" in text
    assert "窗口内完成 12 条" in text


def test_render_text_survives_model_error() -> None:
    module = _load_module()
    text = module.render_text(
        {
            "window_hours": 6,
            "decisions": {"in_window": 0, "executed": 0},
            "shadow": {},
            "model": {"error": "RuntimeError: boom"},
        }
    )

    assert "读取失败" in text
    assert "无阻塞记录" in text
