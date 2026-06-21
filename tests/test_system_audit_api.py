from __future__ import annotations

from typing import Any, Awaitable, Callable

import pytest

from web_dashboard.api import system_audit

AuditFactory = Callable[[], Awaitable[dict[str, Any]]]


def _async_card(
    key: str,
    status: str,
    summary: str,
    *,
    title: str | None = None,
    evidence_value: int = 1,
) -> AuditFactory:
    async def factory() -> dict[str, Any]:
        return system_audit._audit_card(
            key,
            title or key,
            status,
            summary,
            evidence=[{"label": "样本", "value": evidence_value}],
            next_actions=[f"处理 {key}"],
        )

    return factory


@pytest.mark.asyncio
async def test_system_audit_status_aggregates_root_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_trade_loop_audit",
        _async_card("trade_loop", "critical", "交易主循环卡住", title="交易闭环"),
    )
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_audit",
        _async_card("okx_reconciliation", "ok", "OKX 对账正常", title="OKX 历史对账"),
    )
    monkeypatch.setattr(
        system_audit,
        "_position_price_integrity_audit",
        _async_card("position_price_integrity", "ok", "持仓价格一致", title="持仓价格一致性"),
    )
    monkeypatch.setattr(
        system_audit,
        "_market_data_audit",
        _async_card("market_data", "warning", "K线过期", title="行情与 K线"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_quality_audit",
        _async_card("strategy_quality", "ok", "策略质量正常", title="策略质量"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_training_audit",
        _async_card("model_training", "warning", "模型未就绪", title="模型与训练"),
    )

    payload = await system_audit.system_audit_status()

    assert payload["status"] == "critical"
    assert payload["status_label"] == "异常"
    assert payload["summary"] == {
        "cards": 6,
        "critical": 1,
        "warning": 2,
        "ok": 3,
        "findings": 3,
    }
    assert [card["status"] for card in payload["cards"]] == [
        "critical",
        "warning",
        "warning",
        "ok",
        "ok",
        "ok",
    ]
    assert [item["key"] for item in payload["root_causes"]] == [
        "trade_loop",
        "market_data",
        "model_training",
    ]
    assert "只读巡检" in payload["safety_note"]
    assert "人工确认" in payload["safety_note"]


@pytest.mark.asyncio
async def test_system_audit_status_wraps_failed_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_audit() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(system_audit, "_trade_loop_audit", failed_audit)
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_audit",
        _async_card("okx_reconciliation", "ok", "OKX 对账正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_position_price_integrity_audit",
        _async_card("position_price_integrity", "ok", "持仓价格一致"),
    )
    monkeypatch.setattr(
        system_audit,
        "_market_data_audit",
        _async_card("market_data", "ok", "行情正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_quality_audit",
        _async_card("strategy_quality", "ok", "策略质量正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_training_audit",
        _async_card("model_training", "ok", "模型正常"),
    )

    payload = await system_audit.system_audit_status()

    assert payload["status"] == "warning"
    assert payload["summary"]["warning"] == 1
    assert payload["root_causes"][0]["title"] == "巡检模块"
    assert payload["root_causes"][0]["severity"] == "warning"
    assert payload["cards"][0]["details"]["error"]
