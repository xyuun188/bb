from datetime import timedelta

import pytest
from sqlalchemy import select

from config.settings import settings
from core.market_facts import MARKET_FACT_SCHEMA_VERSION
from core.training_contracts import SHADOW_LABEL_VERSION
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from services import ml_signal_service, ml_training_dataset


def test_ml_training_dataset_owns_database_access_functions() -> None:
    owned = {
        "load_shadow_training_rows",
        "count_shadow_training_rows",
        "count_shadow_training_decision_groups",
        "load_authoritative_trade_training_samples",
        "probe_authoritative_trade_training_cursor",
        "select_shadow_training_rows",
    }

    assert owned <= ml_training_dataset.__dict__.keys()
    assert owned.isdisjoint(ml_signal_service.__dict__)


@pytest.mark.asyncio
async def test_crypto_training_scope_requires_frozen_asset_classification() -> None:
    filters = ml_training_dataset.crypto_shadow_candidate_filters()
    query = select(ShadowBacktest.id).where(*filters).compile()
    assert "training_feature_snapshot" in str(query)
    assert "inst_category" in query.params.values()


@pytest.mark.asyncio
async def test_training_scope_counts_only_frozen_crypto_asset_class(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    database_path = tmp_path / "training-asset-class.db"
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{database_path.as_posix()}",
    )
    await init_db()
    created_at = ml_training_dataset.load_training_data_start() + timedelta(minutes=1)

    def sample(symbol: str, category: str | None, index: int) -> ShadowBacktest:
        spec = {"inst_category": category} if category is not None else {}
        return ShadowBacktest(
            decision_id=index,
            model_name="test",
            execution_mode="paper",
            symbol=symbol,
            analysis_type="market",
            decision_action="hold",
            decision_confidence=0.5,
            entry_price=100.0,
            feature_snapshot={
                "market_fact": {
                    "schema_version": MARKET_FACT_SCHEMA_VERSION,
                    "native_identity": {
                        "inst_id": symbol.replace("/", "-") + "-SWAP",
                        "inst_type": "SWAP",
                        "uly": symbol.replace("/", "-"),
                        "contract_spec": spec,
                    },
                },
            },
            status="completed",
            label_version=SHADOW_LABEL_VERSION,
            horizon_minutes=5,
            long_return_pct=-0.1 if category == "1" else 1.0,
            short_return_pct=0.1,
            created_at=created_at,
            due_at=created_at,
        )

    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    sample("BTC/USDT", "1", 1),
                    sample("SOXL/USDT", "3", 2),
                    sample("UNKNOWN/USDT", None, 3),
                ]
            )
            await session.commit()

        filters = ml_training_dataset.crypto_shadow_candidate_filters()
        async with get_session_ctx() as session:
            selected = (
                await session.execute(
                    select(ShadowBacktest.symbol).where(*filters)
                )
            ).scalars().all()

        assert selected == ["BTC/USDT"]
        assert await ml_training_dataset.count_shadow_training_rows() == 1
        assert await ml_training_dataset.count_shadow_training_decision_groups() == 1
    finally:
        await close_db()
