from __future__ import annotations

from services.execution_reason_localizer import localize_execution_reason


def test_localize_profit_first_defensive_probe_reason() -> None:
    reason = (
        "Profit-First defensive probe guard: low-payoff tiny/probe entry was capped to "
        "1x by risk budget and has too little expected profit; keep it shadow-only until "
        "quality upgrades."
    )

    localized = localize_execution_reason(reason)

    assert localized is not None
    assert "Profit-First 防御探针拦截" in localized
    assert "low-payoff" not in localized
    assert "shadow-only" not in localized


def test_localize_unknown_reason_preserves_text() -> None:
    assert localize_execution_reason("自定义中文原因") == "自定义中文原因"


def test_localize_okx_attention_reason_with_dynamic_count() -> None:
    reason = (
        "OKX auto reconciliation found 2 current-state differences requiring review; "
        "pause new entries until reconciled."
    )

    localized = localize_execution_reason(reason)

    assert localized == "OKX 自动对账发现 2 个当前状态差异需要复核；暂停新开仓，等待状态对齐后再恢复。"
    assert "current-state" not in localized


def test_localize_okx_reconciliation_reason_with_dynamic_error() -> None:
    reason = (
        "OKX auto reconciliation is unhealthy: OKX timeout; pause new entries until "
        "OKX/backend state is consistent."
    )

    localized = localize_execution_reason(reason)

    assert localized == "OKX 自动对账异常：OKX timeout；暂停新开仓，等待 OKX 与本地后台状态恢复一致。"
    assert "pause new entries" not in localized
