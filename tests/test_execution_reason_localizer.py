from __future__ import annotations

from services.execution_reason_localizer import localize_execution_reason


def test_localize_unknown_reason_preserves_text() -> None:
    assert localize_execution_reason("自定义中文原因") == "自定义中文原因"


def test_localize_dynamic_exit_dashboard_codes() -> None:
    assert localize_execution_reason("dynamic_exit_policy_passed") == (
        "动态退出策略初步检查已通过，正在继续校验减仓比例、费用和交易规则。"
    )
    assert localize_execution_reason("dynamic_exit_fraction_below_execution_minimum") == (
        "建议减仓比例低于系统最小自动减仓比例 5%，本轮不提交平仓订单，继续持有。"
    )
    assert localize_execution_reason("profit_lock_target_already_filled") == (
        "本轮盈利锁定目标已经由交易所确认成交，无需重复提交平仓订单。"
    )
    assert localize_execution_reason("dynamic_exit_risk_target_already_realized") == (
        "本轮风险减仓目标此前已经完成，无需重复提交平仓订单。"
    )


def test_localize_multiple_dynamic_exit_codes() -> None:
    localized = localize_execution_reason(
        "dynamic_exit_fraction_below_execution_minimum,minimum_position_observation_not_elapsed"
    )

    assert localized == (
        "建议减仓比例低于系统最小自动减仓比例 5%，本轮不提交平仓订单，继续持有；"
        "持仓观察时间尚未达到最小要求，本轮继续观察。"
    )


def test_localize_legacy_final_override_observation_note() -> None:
    localized = localize_execution_reason(
        "Production permission belongs to the authoritative fee-after return policy; "
        "the legacy final decision-maker override is removed."
    )

    assert localized == (
        "生产交易许可由权威费后收益策略统一决定；旧版“最终交易员”强制覆盖权限"
        "已移除，本模型结果仅供观察。"
    )


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
