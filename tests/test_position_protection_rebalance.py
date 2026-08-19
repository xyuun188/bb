from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services import position_protection_rebalance
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
        stuck_algo_id: str = "",
        ignored_amend_algo_id: str = "",
        created_visibility_delay_reads: int = 0,
        lot_size: str = "1",
    ) -> None:
        self.position_contracts = position_contracts
        self.protection_contracts = list(protection_contracts)
        self.protection_algo_ids = [
            f"algo-{index}" for index in range(1, len(self.protection_contracts) + 1)
        ]
        self.next_algo_index = len(self.protection_contracts) + 1
        self.fail_algo_id = fail_algo_id
        self.stuck_algo_id = stuck_algo_id
        self.ignored_amend_algo_id = ignored_amend_algo_id
        self.created_visibility_delay_reads = created_visibility_delay_reads
        self.lot_size = lot_size
        self.hidden_created_reads: dict[str, int] = {}
        self.amend_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []

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
        visible: list[tuple[str, str]] = []
        for algo_id, contracts in zip(
            self.protection_algo_ids,
            self.protection_contracts,
            strict=True,
        ):
            hidden_reads = self.hidden_created_reads.get(algo_id, 0)
            if hidden_reads > 0:
                self.hidden_created_reads[algo_id] = hidden_reads - 1
                continue
            visible.append((algo_id, contracts))
        return [
            {
                "symbol": "IRYS/USDT",
                "position_side": "short",
                "algo_id": algo_id,
                "contracts": contracts,
                "reduce_only": True,
                "state": "live",
                "order_type": "oco",
                "stop_loss_price": 0.16,
                "take_profit_price": 0.14,
                "created_at_ms": index,
                "raw": {"info": {"instId": "IRYS-USDT-SWAP"}},
            }
            for index, (algo_id, contracts) in enumerate(visible, start=1)
        ]

    async def get_open_orders_strict(self, _symbol: str | None) -> list[dict[str, Any]]:
        return []

    async def get_contract_specs_strict(
        self,
        _symbols: list[str],
    ) -> dict[str, dict[str, str]]:
        return {
            "IRYS-USDT-SWAP": {"lotSz": self.lot_size, "minSz": self.lot_size}
        }

    async def amend_position_protection_size(
        self,
        *,
        inst_id: str,
        algo_id: str,
        contracts: float,
    ) -> dict[str, Any]:
        self.amend_calls.append({"inst_id": inst_id, "algo_id": algo_id, "contracts": contracts})
        if algo_id == self.stuck_algo_id:
            raise RuntimeError(
                "OKX API error [51513]: Number of modification requests in progress exceeded"
            )
        if algo_id == self.fail_algo_id:
            return {"code": "1", "data": [{"algoId": algo_id, "sCode": "51000"}]}
        if algo_id == self.ignored_amend_algo_id:
            return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}
        index = self.protection_algo_ids.index(algo_id)
        self.protection_contracts[index] = str(contracts).removesuffix(".0")
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}

    async def cancel_position_protection_order(
        self,
        *,
        inst_id: str,
        algo_id: str,
    ) -> dict[str, Any]:
        self.cancel_calls.append({"inst_id": inst_id, "algo_id": algo_id})
        index = self.protection_algo_ids.index(algo_id)
        self.protection_algo_ids.pop(index)
        self.protection_contracts.pop(index)
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}

    async def create_position_protection_order(
        self,
        *,
        inst_id: str,
        position_side: str,
        okx_position_side: str,
        contracts: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> dict[str, Any]:
        self.create_calls.append(
            {
                "inst_id": inst_id,
                "position_side": position_side,
                "okx_position_side": okx_position_side,
                "contracts": contracts,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
            }
        )
        self.protection_contracts.append(str(contracts).removesuffix(".0"))
        algo_id = f"algo-{self.next_algo_index}"
        self.next_algo_index += 1
        self.protection_algo_ids.append(algo_id)
        self.hidden_created_reads[algo_id] = self.created_visibility_delay_reads
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
async def test_position_increase_adds_delta_oco_without_replacing_existing_order() -> None:
    executor = _Executor(position_contracts="10", protection_contracts=("1",))

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert executor.protection_contracts == ["1", "9"]
    assert executor.amend_calls == []
    assert executor.create_calls == [
        {
            "inst_id": "IRYS-USDT-SWAP",
            "position_side": "short",
            "okx_position_side": "net",
            "contracts": 9.0,
            "stop_loss_price": 0.16,
            "take_profit_price": 0.14,
        }
    ]


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
async def test_stuck_okx_amend_is_replaced_without_unprotected_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="5",
        protection_contracts=("13",),
        stuck_algo_id="algo-1",
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert result["status"] == "repaired"
    assert executor.protection_contracts == ["5"]
    assert executor.create_calls == [
        {
            "inst_id": "IRYS-USDT-SWAP",
            "position_side": "short",
            "okx_position_side": "net",
            "contracts": 5.0,
            "stop_loss_price": 0.16,
            "take_profit_price": 0.14,
        }
    ]
    assert executor.cancel_calls == [{"inst_id": "IRYS-USDT-SWAP", "algo_id": "algo-1"}]
    assert result["applied_actions"][0]["action"]["action"] == "replace_stuck_amend"


@pytest.mark.asyncio
async def test_acknowledged_but_unapplied_amend_uses_verified_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="5",
        protection_contracts=("13",),
        ignored_amend_algo_id="algo-1",
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert result["status"] == "repaired"
    assert result["fallback_reason"] == "okx_amend_acknowledged_but_not_observed"
    assert result["amend_verification_attempts"] == 4
    assert executor.protection_contracts == ["5"]
    assert executor.create_calls == [
        {
            "inst_id": "IRYS-USDT-SWAP",
            "position_side": "short",
            "okx_position_side": "net",
            "contracts": 5.0,
            "stop_loss_price": 0.16,
            "take_profit_price": 0.14,
        }
    ]
    assert executor.cancel_calls == [{"inst_id": "IRYS-USDT-SWAP", "algo_id": "algo-1"}]
    assert result["applied_actions"][-1]["action"]["action"] == "replace_stuck_amend"


@pytest.mark.asyncio
async def test_stuck_split_oco_replacement_is_created_and_verified_as_one_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="5",
        protection_contracts=("6", "7"),
        stuck_algo_id="algo-1",
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert executor.protection_contracts == ["2", "3"]
    assert len(executor.create_calls) == 2
    assert {call["algo_id"] for call in executor.cancel_calls} == {"algo-1", "algo-2"}
    assert len(result["applied_actions"]) == 2


@pytest.mark.asyncio
async def test_mixed_resize_and_delta_plan_replaces_complete_group_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="3.44",
        protection_contracts=("1.6", "0.57", "0.93", "0.34", "0.03", "0.03", "0.02", "0.04"),
        lot_size="0.01",
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert result["fallback_reason"] == "mixed_protection_plan_create_before_cancel"
    assert executor.amend_calls == []
    assert len(executor.create_calls) == 9
    assert len(executor.cancel_calls) == 8
    assert all(call["algo_id"] for call in executor.cancel_calls)
    assert sum(float(value) for value in executor.protection_contracts) == pytest.approx(3.44)
    assert result["after"]["coverage_mismatches"] == []


@pytest.mark.asyncio
async def test_mixed_replacement_drops_below_minimum_slice_without_zero_size_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="3.0",
        protection_contracts=(
            "0.04",
            "1.54",
            "0.55",
            "0.89",
            "0.32",
            "0.02",
            "0.02",
            "0.01",
            "0.05",
        ),
        lot_size="0.01",
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    assert result["fallback_reason"] == "mixed_protection_plan_create_before_cancel"
    assert executor.amend_calls == []
    assert len(executor.create_calls) == 9
    assert all(call["contracts"] >= 0.01 for call in executor.create_calls)
    assert len(executor.cancel_calls) == 9
    assert sum(float(value) for value in executor.protection_contracts) == pytest.approx(3.0)
    assert result["after"]["coverage_mismatches"] == []


@pytest.mark.asyncio
async def test_replacement_waits_for_delayed_okx_algo_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(position_protection_rebalance, "PROTECTION_VERIFY_DELAY_SECONDS", 0)
    executor = _Executor(
        position_contracts="5",
        protection_contracts=("13",),
        ignored_amend_algo_id="algo-1",
        created_visibility_delay_reads=6,
    )

    result = await rebalance_position_protection_after_exit(executor, _decision())

    assert result["verified"] is True
    replacement = result["applied_actions"][-1]
    assert replacement["create_verification_attempts"] == 7
    assert executor.protection_contracts == ["5"]


@pytest.mark.asyncio
async def test_open_position_without_protection_fails_closed_without_mutation() -> None:
    executor = _Executor(position_contracts="5", protection_contracts=())

    with pytest.raises(PositionProtectionRebalanceError) as caught:
        await rebalance_position_protection_after_exit(executor, _decision())

    assert caught.value.report["status"] == "blocked"
    assert caught.value.report["before"]["missing_keys"] == [["IRYS/USDT", "short"]]
    assert executor.amend_calls == []
