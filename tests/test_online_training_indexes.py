from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ensure_online_reconciliation_indexes import (
    INDEX_DDLS,
    TRAINING_INDEX_DDLS,
    _apply_indexes,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("validity", (None, True, False))
@pytest.mark.parametrize("index", TRAINING_INDEX_DDLS)
async def test_training_index_recovers_only_failed_concurrent_build(validity, index) -> None:
    commands: list[str] = []

    class Connection:
        async def execute(self, statement, parameters=None):
            commands.append(str(statement))
            if parameters is not None:
                assert parameters == {"index_name": index[0]}
            return SimpleNamespace(scalar=lambda: validity)

    await _apply_indexes(Connection(), (index,), rollback=False)

    drops = [command for command in commands if command.startswith("DROP INDEX")]
    assert len(drops) == int(validity is False)
    assert commands[-1] == index[1]
    assert "CREATE INDEX CONCURRENTLY" in commands[-1]
    assert "INCLUDE (decision_id)" in commands[-1]
    if "crypto_training" in index[0]:
        assert "inst_category' = '1'" in commands[-1]
        assert "long_return_pct IS NOT NULL AND short_return_pct IS NOT NULL" in commands[-1]
    else:
        assert "historical_market_path_only" in commands[-1]


@pytest.mark.asyncio
async def test_training_index_rollback_does_not_touch_reconciliation_indexes() -> None:
    commands: list[str] = []

    class Connection:
        async def execute(self, statement):
            commands.append(str(statement))

    await _apply_indexes(Connection(), TRAINING_INDEX_DDLS, rollback=True)

    assert commands == [
        f"DROP INDEX CONCURRENTLY IF EXISTS {name}" for name, _ in TRAINING_INDEX_DDLS
    ]
    assert all(name not in command for name, _ in INDEX_DDLS for command in commands)
