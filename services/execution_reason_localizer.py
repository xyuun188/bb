"""Localize current exchange and runtime execution reasons."""

from __future__ import annotations

_EXACT_TRANSLATIONS = {
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
        translated = _localize_count_reason(candidate) or _localize_dynamic_reason(candidate)
        if translated is not None:
            return translated
    return text
