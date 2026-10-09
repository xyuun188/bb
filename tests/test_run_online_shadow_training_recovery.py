from __future__ import annotations

import argparse

from scripts.run_online_shadow_training_recovery import _should_manage_trading_service


def _args(*, training_only: bool = False, counts_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        training_only=training_only,
        counts_only=counts_only,
    )


def test_training_only_does_not_stop_or_restart_trading_service() -> None:
    assert _should_manage_trading_service(_args(training_only=True)) is False


def test_counts_only_does_not_stop_or_restart_trading_service() -> None:
    assert _should_manage_trading_service(_args(counts_only=True)) is False


def test_full_historical_rebuild_manages_trading_service() -> None:
    assert _should_manage_trading_service(_args()) is True
