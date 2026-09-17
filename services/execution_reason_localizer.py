"""Localize current exchange and runtime execution reasons."""

from __future__ import annotations

_EXACT_TRANSLATIONS = {
    "前一笔平仓已经完成，当前没有可平仓位；本次重复平仓请求已跳过，未再次提交交易所。": "前一笔平仓已经完成，当前没有可平仓位；本次重复平仓请求已跳过，未再次提交交易所。",
    "no_local_position": "前一笔平仓已经完成，当前没有可平仓位；本次重复平仓请求已跳过，未再次提交交易所。",
    "dynamic_exit_policy_passed": "动态退出策略初步检查已通过，正在继续校验减仓比例、费用和交易规则。",
    "dynamic_exit_fraction_below_execution_minimum": "建议减仓比例低于系统最小自动减仓比例 5%，本轮不提交平仓订单，继续持有。",
    "dynamic_exit_pressure_zero": "当前没有达到减仓或平仓条件，本轮继续持有。",
    "dynamic_exit_target_already_realized": "计划减仓目标此前已经完成，本轮无需重复提交平仓订单。",
    "dynamic_exit_policy_not_eligible": "当前动态退出条件未满足，本轮不提交平仓订单。",
    "dynamic_exit_close_fraction_not_positive": "本轮建议减仓比例为 0，不提交平仓订单。",
    "dynamic_exit_fraction_zero": "本轮建议减仓比例为 0，不提交平仓订单。",
    "position_economics_missing": "缺少当前仓位的完整成本和盈亏数据，暂不自动减仓。",
    "current_position_management_contract_incomplete": "当前仓位管理数据不完整，暂不自动减仓。",
    "fee_after_profit_not_positive": "扣除手续费、滑点和资金费后，当前退出收益不为正，本轮继续持有。",
    "exit_execution_cost_missing": "缺少完整的退出手续费或滑点数据，暂不自动减仓。",
    "position_age_evidence_missing": "缺少可靠的持仓时间数据，暂不自动减仓。",
    "minimum_position_observation_not_elapsed": "持仓观察时间尚未达到最小要求，本轮继续观察。",
    "early_exit_observation_active": "仓位仍处于开仓后的早期观察期，本轮暂不自动减仓。",
    "economic_exit_evidence_incomplete": "退出所需的收益和成本证据不完整，暂不自动减仓。",
    "expected_net_return_not_positive": "预计扣除交易成本后的收益不为正，本轮不开仓。",
    "authoritative_fee_after_return_lcb_not_positive": "权威费后收益分布下界不为正，本轮不开仓。",
    "entry_evidence_wait": "开仓证据尚未准备完整，本轮继续等待。",
    "model_timeout": "模型调用超时，本轮没有生成可执行决策。",
    "no_candidate": "本轮没有发现满足开仓条件的候选机会。",
    "Production permission belongs to the authoritative fee-after return policy; the legacy final decision-maker override is removed.": "生产交易许可由权威费后收益策略统一决定；旧版“最终交易员”强制覆盖权限已移除，本模型结果仅供观察。",
    "OKX auto reconciliation is unhealthy; pause new entries until OKX/backend state is consistent.": "OKX 自动对账异常；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    "OKX auto reconciliation is stale; pause new entries until OKX/backend state is consistent.": "OKX 自动对账已过期；暂停新开仓，等待 OKX 与本地后台状态恢复一致。",
    "Trading runtime heartbeat is unavailable; new entries are blocked until the runtime publishes a fresh OKX sync heartbeat.": "交易运行时心跳不可用；暂停新开仓，直到运行时发布新的 OKX 同步心跳。",
    "OKX runtime sync healthy for new entries.": "OKX 运行态同步正常，允许新开仓。",
    "Trading runtime is not running; OKX runtime sync cannot authorize new entries.": "交易运行时未运行；OKX 运行态同步无法授权新开仓。",
    "Trading runtime heartbeat is stale; new entries are blocked until a fresh OKX sync heartbeat is observed.": "交易运行时心跳已过期；暂停新开仓，直到观察到新的 OKX 同步心跳。",
    "OKX runtime sync is stale; new entries are blocked.": "OKX 运行态同步已过期；暂停新开仓。",
    "OKX runtime sync is unhealthy; new entries are blocked.": "OKX 运行态同步异常；暂停新开仓。",
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
        if text.startswith(prefix) and text.endswith(suffix):
            count = text.removeprefix(prefix).removesuffix(suffix).strip()
            if count.isdigit():
                return template.format(count=count)
    return None


def _localize_dynamic_reason(text: str) -> str | None:
    for prefix, suffix, template in _DYNAMIC_TRANSLATIONS:
        if not text.startswith(prefix) or (suffix and not text.endswith(suffix)):
            continue
        detail = text.removeprefix(prefix)
        if suffix:
            detail = detail.removesuffix(suffix)
        if detail.strip():
            return template.format(detail=detail.strip())
    return None


def _localize_reason_code_list(text: str) -> str | None:
    codes = [item.strip() for item in text.split(",") if item.strip()]
    if not codes or any(code not in _EXACT_TRANSLATIONS for code in codes):
        return None
    translated = [_EXACT_TRANSLATIONS[code] for code in codes]
    sentences = [item.rstrip("。") for item in dict.fromkeys(translated)]
    return "；".join(sentences) + "。"


def localize_execution_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    text = str(reason).strip()
    if not text:
        return text
    for candidate in (text, " ".join(text.split())):
        translated = _EXACT_TRANSLATIONS.get(candidate)
        if translated is not None:
            return translated
        translated = (
            _localize_reason_code_list(candidate)
            or _localize_count_reason(candidate)
            or _localize_dynamic_reason(candidate)
        )
        if translated is not None:
            return translated
    return text
