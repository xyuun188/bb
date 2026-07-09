from ai_brain.base_model import Action, DecisionOutput
from services.exit_position_snapshot import ExitPositionSnapshotPolicy


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="测试平仓",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
    )


class FakeSyncService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def reconcile_positions(self, reason: str, **kwargs) -> None:
        self.calls.append(("reconcile", {"reason": reason, **kwargs}))

    async def get_open_positions_context(self) -> list[dict]:
        self.calls.append(("context", ""))
        return [{"symbol": "BTC/USDT", "side": "long", "quantity": 1}]

    async def has_matching_exchange_exit_position(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool | None:
        self.calls.append((model_name, decision.symbol))
        return None


async def test_exit_position_snapshot_refreshes_and_replaces_open_positions() -> None:
    sync = FakeSyncService()
    policy = ExitPositionSnapshotPolicy(sync)
    open_positions = [{"symbol": "OLD/USDT"}]

    positions = await policy.refresh_positions(open_positions)

    assert positions == [{"symbol": "BTC/USDT", "side": "long", "quantity": 1}]
    assert open_positions == positions
    assert sync.calls == [("context", "")]


async def test_exit_position_snapshot_can_run_bounded_reconcile_when_enabled() -> None:
    sync = FakeSyncService()
    policy = ExitPositionSnapshotPolicy(sync, reconcile_timeout_seconds=1.5)
    open_positions = [{"symbol": "OLD/USDT"}]

    positions = await policy.refresh_positions(open_positions)

    assert positions == [{"symbol": "BTC/USDT", "side": "long", "quantity": 1}]
    assert sync.calls == [
        (
            "reconcile",
            {
                "reason": "exit precheck",
                "timeout_seconds": 1.5,
                "lock_wait_seconds": 0.05,
                "record_timeout_error": False,
            },
        ),
        ("context", ""),
    ]


async def test_exit_position_snapshot_proxies_exchange_match_status() -> None:
    sync = FakeSyncService()
    policy = ExitPositionSnapshotPolicy(sync)

    result = await policy.has_matching_exchange_position(
        "ensemble_trader",
        _decision(),
    )

    assert result is None
    assert sync.calls == [("ensemble_trader", "BTC/USDT")]
