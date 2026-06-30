import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.exit_profit_precheck import ExitProfitPrecheckPolicy


def _normalize_symbol(value) -> str:
    return str(value or "").replace("/", "-").replace("-SWAP", "")


def _decision(raw_response: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="测试锁盈",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response=raw_response
        or {
            "close_evidence": {"profit_protection": True},
        },
    )


def _position(**overrides) -> dict:
    data = {
        "model_name": "ensemble_trader",
        "symbol": "BTC-USDT-SWAP",
        "side": "long",
        "quantity": 1.0,
        "entry_price": 100.0,
    }
    data.update(overrides)
    return data


def _policy(latest_price: float) -> ExitProfitPrecheckPolicy:
    async def latest_price_provider(symbol: str) -> float:
        return latest_price

    return ExitProfitPrecheckPolicy(latest_price_provider, _normalize_symbol)


@pytest.mark.asyncio
async def test_exit_profit_precheck_blocks_when_latest_price_missing() -> None:
    reason = await _policy(0.0).guard_reason(_decision(), [_position()])

    assert reason == "利润保护平仓前未能重新获取最新价格，系统不使用过期浮盈判断执行锁盈单。"


@pytest.mark.asyncio
async def test_exit_profit_precheck_blocks_pure_lock_profit_when_fresh_profit_is_too_small() -> (
    None
):
    decision = _decision()

    reason = await _policy(100.2).guard_reason(decision, [_position()])

    assert reason is not None
    assert "利润保护执行前复核未通过" in reason
    guard = decision.raw_response["execution_profit_protection_guard"]
    assert guard["applied"] is True
    assert guard["estimated_unrealized_pnl"] == 0.2
    assert guard["min_required_profit"] == 0.75


@pytest.mark.asyncio
async def test_exit_profit_precheck_allows_non_profit_risk_exit_evidence() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "profit_protection": True,
                "hard_risk": True,
            }
        }
    )

    reason = await _policy(100.2).guard_reason(decision, [_position()])

    assert reason is None
    guard = decision.raw_response["execution_profit_protection_guard"]
    assert guard["applied"] is False
    assert guard["non_profit_exit_evidence"] is True


@pytest.mark.asyncio
async def test_exit_profit_precheck_allows_structured_predictive_downside_intent() -> None:
    decision = _decision(
        {
            "exit_intent": "predictive_downside",
            "close_evidence": {"profit_protection": True},
        }
    )

    reason = await _policy(100.2).guard_reason(decision, [_position()])

    assert reason is None
    guard = decision.raw_response["execution_profit_protection_guard"]
    assert guard["non_profit_exit_evidence"] is True


@pytest.mark.asyncio
async def test_exit_profit_precheck_uses_small_position_profit_floor() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "profit_protection": True,
                "small_position_profit_lock": True,
            }
        }
    )

    reason = await _policy(101.2).guard_reason(decision, [_position(quantity=0.5)])

    assert reason is None
    assert "execution_profit_protection_guard" not in decision.raw_response


@pytest.mark.asyncio
async def test_exit_profit_precheck_allows_profitable_fresh_recheck() -> None:
    reason = await _policy(102.0).guard_reason(_decision(), [_position()])

    assert reason is None
