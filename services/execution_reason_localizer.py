"""Localize stored execution reasons for operator-facing surfaces."""

from __future__ import annotations


_EXACT_TRANSLATIONS = {
    (
        "Profit-First defensive probe guard: low-payoff tiny/probe entry was capped to "
        "1x by risk budget and has too little expected profit; keep it shadow-only until "
        "quality upgrades."
    ): (
        "Profit-First 防御探针拦截：该极小/探针开仓属于低收益质量，又被风险预算限制为 "
        "1 倍杠杆，预期实际盈利过低；本轮只记录影子样本，等收益质量升级后再允许真实开仓。"
    ),
    (
        "Profit-First v3 probe-loss brake: recent tiny/probe closes are all losing; "
        "this candidate stays shadow unless it upgrades to validated_probe or better."
    ): (
        "Profit-First 探针亏损刹车：最近的极小/探针仓位平仓全部亏损；"
        "该候选先保留为影子样本，只有升级到已验证探针或更高质量档后才允许真实开仓。"
    ),
    (
        "Profit-First v3 release net-benefit guard: this is a losing release and there "
        "is no hard risk or stronger replacement opportunity; keep it open for the next review."
    ): (
        "Profit-First 释放净收益保护：当前是亏损释放信号，且没有硬风险或更强替代机会；"
        "本轮先保留仓位，等待下一轮复盘。"
    ),
    (
        "Profit-First trade plan is incomplete or shadow-only; entry stayed in shadow before "
        "OKX submit."
    ): "Profit-First 交易计划不完整或仍处于影子档，本轮未提交 OKX 开仓订单。",
    (
        "Profit-First entry has no profit_risk_sizing snapshot; entry blocked before OKX submit."
    ): "Profit-First 开仓缺少收益/风险仓位快照，本轮在提交 OKX 前拦截。",
    (
        "Profit-First position ladder was missing and late reconstruction would change size; "
        "entry blocked before OKX submit."
    ): "Profit-First 仓位阶梯缺失，且临时重建会改变下单仓位；本轮在提交 OKX 前拦截。",
    (
        "Profit-First position ladder is shadow-only; entry stayed in shadow before OKX submit."
    ): "Profit-First 仓位阶梯仍为影子档，本轮未提交 OKX 开仓订单。",
    (
        "Profit-First position ladder produced zero real size; entry blocked before OKX submit."
    ): "Profit-First 仓位阶梯计算出的真实仓位为 0，本轮在提交 OKX 前拦截。",
    (
        "Entry evidence explicitly recommends hold or tiny shadow probe only."
    ): "入场证据明确建议观望或只保留极小影子探针，本轮不提交 OKX 订单。",
    (
        "Entry evidence was retained for shadow learning, but it did not meet the controlled "
        "probe conversion thresholds."
    ): "入场证据已保留用于影子学习，但未达到受控探针转换阈值，本轮不提交 OKX 订单。",
    (
        "Entry evidence score is below the controlled probe floor."
    ): "入场证据分数低于受控探针底线，本轮不提交 OKX 订单。",
    (
        "OKX auto reconciliation is unhealthy; pause new entries until OKX/backend state is consistent."
    ): "OKX 自动对账异常；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    (
        "OKX auto reconciliation is stale; pause new entries until OKX/backend state is consistent."
    ): "OKX 自动对账已过期；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    (
        "Trading runtime heartbeat is unavailable; new entries are blocked until the runtime "
        "publishes a fresh OKX sync heartbeat."
    ): "交易运行时心跳不可用；暂停新开仓，直到运行时发布新的 OKX 同步心跳。",
    (
        "OKX runtime sync healthy for new entries."
    ): "OKX 运行态同步正常，允许新开仓。",
    (
        "Trading runtime is not running; OKX runtime sync cannot authorize new entries."
    ): "交易运行时未运行；OKX 运行态同步无法授权新开仓。",
    (
        "Trading runtime heartbeat is stale; new entries are blocked until a fresh OKX sync "
        "heartbeat is observed."
    ): "交易运行时心跳已过期；暂停新开仓，直到观察到新的 OKX 同步心跳。",
    (
        "OKX runtime sync is stale; new entries are blocked."
    ): "OKX 运行态同步已过期；暂停新开仓。",
    (
        "OKX runtime sync is unhealthy; new entries are blocked."
    ): "OKX 运行态同步异常；暂停新开仓。",
}

_PREFIX_TRANSLATIONS = (
    (
        "OKX auto reconciliation found ",
        " current-state differences requiring review; pause new entries until reconciled.",
        "OKX 自动对账发现 {count} 个当前状态差异需要复核；暂停新开仓，等待状态对齐后再恢复。",
    ),
    (
        "OKX runtime sync found ",
        " current-state differences; new entries are blocked until reconciled.",
        "OKX 运行态同步发现 {count} 个当前状态差异；暂停新开仓，等待状态对齐后再恢复。",
    ),
)

_DYNAMIC_TRANSLATIONS = (
    (
        "OKX auto reconciliation is unhealthy: ",
        "; pause new entries until OKX/backend state is consistent.",
        "OKX 自动对账异常：{detail}；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    ),
    (
        "OKX auto reconciliation is stale: ",
        "; pause new entries until OKX/backend state is consistent.",
        "OKX 自动对账已过期：{detail}；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    ),
    (
        "OKX runtime sync is unhealthy; new entries are blocked. Last error: ",
        "",
        "OKX 运行态同步异常；暂停新开仓。最近错误：{detail}",
    ),
    (
        "OKX runtime sync is stale; new entries are blocked. Last error: ",
        "",
        "OKX 运行态同步已过期；暂停新开仓。最近错误：{detail}",
    ),
)


def _localize_count_reason(text: str) -> str | None:
    for prefix, suffix, template in _PREFIX_TRANSLATIONS:
        if not text.startswith(prefix) or not text.endswith(suffix):
            continue
        count = text.removeprefix(prefix).removesuffix(suffix).strip()
        if count.isdigit():
            return template.format(count=count)
    return None


def _localize_dynamic_reason(text: str) -> str | None:
    for prefix, suffix, template in _DYNAMIC_TRANSLATIONS:
        if not text.startswith(prefix):
            continue
        if suffix and not text.endswith(suffix):
            continue
        detail = text.removeprefix(prefix)
        if suffix:
            detail = detail.removesuffix(suffix)
        detail = detail.strip()
        if detail:
            return template.format(detail=detail)
    return None


def localize_execution_reason(reason: str | None) -> str | None:
    """Return a Chinese operator-facing reason while preserving unknown text."""

    if reason is None:
        return None
    text = str(reason).strip()
    if not text:
        return text
    translated = _EXACT_TRANSLATIONS.get(text)
    if translated is not None:
        return translated
    translated = _localize_count_reason(text)
    if translated is not None:
        return translated
    translated = _localize_dynamic_reason(text)
    if translated is not None:
        return translated
    normalized = " ".join(text.split())
    translated = _EXACT_TRANSLATIONS.get(normalized)
    if translated is not None:
        return translated
    return _localize_count_reason(normalized) or _localize_dynamic_reason(normalized) or text
