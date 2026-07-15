from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts import repair_okx_protection_integrity as repair


class _SpecsClient:
    async def fetch_contract_specs(self, *, symbols: list[str]) -> dict[str, dict[str, str]]:
        assert symbols == ["IRYS/USDT"]
        return {"IRYS-USDT-SWAP": {"lotSz": "1", "minSz": "1"}}


class _RepairExecutor:
    instances: list[_RepairExecutor] = []
    mutate_position_on_apply = False

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.position_contracts = "13"
        self.protection_contracts = "20"
        self.amend_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.shutdown_called = False
        type(self).instances.append(self)

    async def get_positions_strict(self, _symbol: str | None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "IRYS/USDT",
                "side": "short",
                "contracts": self.position_contracts,
                "info": {
                    "instId": "IRYS-USDT-SWAP",
                    "pos": f"-{self.position_contracts}",
                },
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
                "algo_id": "algo-irys",
                "contracts": self.protection_contracts,
                "reduce_only": True,
                "state": "live",
                "order_type": "oco",
                "stop_loss_price": 0.16,
                "take_profit_price": 0.14,
                "created_at_ms": 1,
                "raw": {"info": {"instId": "IRYS-USDT-SWAP"}},
            }
        ]

    async def get_open_orders_strict(self, _symbol: str | None) -> list[dict[str, Any]]:
        return []

    def _native_facts_client(self) -> _SpecsClient:
        return _SpecsClient()

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
        self.protection_contracts = str(contracts).removesuffix(".0")
        if type(self).mutate_position_on_apply:
            self.position_contracts = "12"
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}

    async def cancel_position_protection_order(
        self,
        *,
        inst_id: str,
        algo_id: str,
    ) -> dict[str, Any]:
        self.cancel_calls.append({"inst_id": inst_id, "algo_id": algo_id})
        return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}

    async def shutdown(self) -> None:
        self.shutdown_called = True


def _args(*, apply: bool, fingerprint: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        mode="paper",
        apply=apply,
        expected_fingerprint=fingerprint,
    )


def _is_file(path: str) -> bool:
    return Path(path).is_file()


def _backup_to(directory: Path):
    def backup(_payload: dict[str, Any]) -> Path:
        path = directory / "protection-backup.json"
        path.write_text("{}", encoding="utf-8")
        return path

    return backup


@pytest.fixture(autouse=True)
def _reset_executor() -> None:
    _RepairExecutor.instances = []
    _RepairExecutor.mutate_position_on_apply = False


@pytest.mark.asyncio
async def test_dry_run_never_mutates_protection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repair, "OKXExecutor", _RepairExecutor)

    result = await repair._run(_args(apply=False))
    executor = _RepairExecutor.instances[-1]

    assert result["apply"] is False
    assert result["before"]["repair_ready"] is True
    assert result["before"]["repair_actions"][0]["action"] == "amend_size"
    assert executor.amend_calls == []
    assert executor.cancel_calls == []
    assert executor.shutdown_called is True


@pytest.mark.asyncio
async def test_apply_rejects_missing_or_stale_dry_run_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair, "OKXExecutor", _RepairExecutor)

    with pytest.raises(RuntimeError, match="requires --expected-fingerprint"):
        await repair._run(_args(apply=True))
    with pytest.raises(RuntimeError, match="changed after dry-run"):
        await repair._run(_args(apply=True, fingerprint="stale"))

    assert all(not executor.amend_calls for executor in _RepairExecutor.instances)


@pytest.mark.asyncio
async def test_apply_backs_up_amends_only_protection_and_verifies_position_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(repair, "OKXExecutor", _RepairExecutor)
    monkeypatch.setattr(repair, "_backup", _backup_to(tmp_path))
    dry_run = await repair._run(_args(apply=False))

    result = await repair._run(
        _args(
            apply=True,
            fingerprint=dry_run["before"]["input_fingerprint"],
        )
    )
    executor = _RepairExecutor.instances[-1]

    assert result["verified"] is True
    assert result["positions_unchanged"] is True
    assert result["verification_errors"] == []
    assert executor.amend_calls == [
        {
            "inst_id": "IRYS-USDT-SWAP",
            "algo_id": "algo-irys",
            "contracts": 13.0,
        }
    ]
    assert executor.cancel_calls == []
    assert _is_file(result["backup_path"])
    assert result["after"]["coverage_mismatches"] == []


@pytest.mark.asyncio
async def test_apply_reports_failure_when_position_changes_during_protection_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(repair, "OKXExecutor", _RepairExecutor)
    monkeypatch.setattr(repair, "_backup", _backup_to(tmp_path))
    dry_run = await repair._run(_args(apply=False))
    _RepairExecutor.mutate_position_on_apply = True

    result = await repair._run(
        _args(
            apply=True,
            fingerprint=dry_run["before"]["input_fingerprint"],
        )
    )

    assert result["verified"] is False
    assert result["positions_unchanged"] is False
    assert "positions_changed_during_protection_repair" in result["verification_errors"]
