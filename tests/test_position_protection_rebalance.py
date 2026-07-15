from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.position_protection_rebalance import (
    PositionProtectionRebalanceError,
    rebalance_position_protection_after_exit,
)


class _Executor:
    def __init__(
        self,
        *,
        position_contracts: str = "5",
        protection_contracts: tuple[str, ...] = ("13",),
        fail_algo_id: str = "",
    ) -> None:
        self.position_contracts = position_contracts
        self.protection_contracts = list(protection_contracts)
        self.fail_algo_id = fail_algo_id
        self.amend_calls: list[dict[str, Any]] = []

    async def get_positions_strict(self, _symbol: str | None) -> list[dict[str, Any]]:
        if not self.position_contracts:
            return []
        return [
            {
                "symbol": "IRYS/USDT",
                "side": "short",
                "contracts": self.position_contracts,
                "info": {"instId": "IRYS-USDT-SWAP", "posSide": "short"},
            }
        ]

    async def get_position_protection_orders(
        self,
        _symbol: str | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "IRYS/USDT",
                "position_side": "short",
                "algo_id": f"algo-{index}",
                "contracts": contracts,
                "reduce_only": True,
                "state": "live",
                "order_type": "oco",
                "stop_loss_price": 0.16,
                "take_profit_price": 0.14,
                "created_at_ms": index,
                "raw": {"info": {"instId": "IRYS-USDT-SWAP"}},
            }
            for index, contracts in enumerate(self.protection_contracts, start=1)
        ]

    async def get_open_orders_strict(self, _symbol: str | None) -> list[dict[str, Any]]:
        return []

    async def get_contract_specs_strict(
        self,
        _symbols: list[str],
    ) -> dict[str, dict[str, str]]:
        return {"IRYS-USDT-SWAP": {"lotSz": "1", "minSz": "1"}}

    async def amend_position_protection_size(
        self,
        *,
        inst_id: str,
        algo_id: str,
        contracts: float,
    ) -> dict[str, Any]:
        self.amend_calls.append(
            {"inst_id": inst_id, "algo_id": algo_id, "contracts": contracts}
        )
        if algo_id == self.fail_algo_id:
            return {"code": "1", "data": [{"algoId": algo_id, "sCode": "51000"}]}
        index = int(algo_id.rsplit("-", 1)[-1]) - 1
        self.protection_contracts[index] = str(contracts).removesuffix(".0")
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}

    async def cancel_position_protection_order(
        self,
        *,
        inst_id: str,
        algo_id: str,
    ) -> dict[str, Any]:
        index = int(algo_id.rsplit("-", 1)[-1]) - 1
        self.protection_contracts.pop(index)
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="IRYS/USDT",
        action=Action.CLOSE_SHORT,
        confidence=0.0,
        reasoning="test",
        position_size_pct=0.5,
    )


@pytest.mark.asyncio
async def test_partial_exit_resizes_split_oco_to_exact_exchange_position() -> None:
    executor = _Executor(position_contracts="5", protection_contracts=("6", "7"))

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert result["status"] == "repaired"
    assert executor.protection_contracts == ["2", "3"]
    assert sum(float(value) for value in executor.protection_contracts) == 5.0
    assert result["after"]["coverage_mismatches"] == []


@pytest.mark.asyncio
async def test_resize_failure_rolls_back_prior_amendment() -> None:
    executor = _Executor(
        position_contracts="5",
        protection_contracts=("6", "7"),
        fail_algo_id="algo-2",
    )

    with pytest.raises(PositionProtectionRebalanceError) as caught:
        await rebalance_position_protection_after_exit(executor, _decision())

    assert caught.value.report["status"] == "apply_failed"
    assert executor.protection_contracts == ["6", "7"]
    assert caught.value.report["rollback_results"][0]["applied"] is True


@pytest.mark.asyncio
async def test_open_position_without_protection_fails_closed_without_mutation() -> None:
    executor = _Executor(position_contracts="5", protection_contracts=())

    with pytest.raises(PositionProtectionRebalanceError) as caught:
        await rebalance_position_protection_after_exit(executor, _decision())

    assert caught.value.report["status"] == "blocked"
    assert caught.value.report["before"]["missing_keys"] == [["IRYS/USDT", "short"]]
    assert executor.amend_calls == []
