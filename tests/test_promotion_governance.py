from __future__ import annotations

import pytest

from core.experiment_contracts import ExperimentContractError
from core.promotion_contracts import (
    build_promotion_review,
    build_runtime_config_snapshot,
    evaluate_promotion_gate,
    verify_promotion_review,
    verify_runtime_config_snapshot,
)
from services.promotion_registry import PromotionRegistry
from services.runtime_config_registry import RuntimeConfigRegistry


def _evidence(**overrides):
    evidence = {
        "metrics": {
            "authoritative_return_lcb_pct": 0.4,
            "authoritative_profit_factor": 1.3,
            "authoritative_trade_count": 50,
        },
        "walk_forward_stability": True,
        "market_regime_stability": True,
        "rolling_distribution_stability": True,
        "return_completeness_verified": True,
        "okx_fact_linkage_verified": True,
    }
    evidence.update(overrides)
    return evidence


def test_live_promotion_is_evidence_gated_and_requires_manual_review() -> None:
    gate = evaluate_promotion_gate(_evidence())
    assert gate["promotion_ready"] is True
    assert gate["decision"] == "manual_review_required"
    assert gate["automatic_live_promotion"] is False

    review = build_promotion_review(
        artifact_id="strategy:v2",
        strategy_id="strategy",
        strategy_version="2",
        from_stage="paper",
        to_stage="live",
        evidence=_evidence(),
    )
    verify_promotion_review(review)

    with pytest.raises(ExperimentContractError, match="passing evidence gate"):
        build_promotion_review(
            artifact_id="strategy:v2",
            strategy_id="strategy",
            strategy_version="2",
            from_stage="paper",
            to_stage="live",
            evidence=_evidence(okx_fact_linkage_verified=False),
        )


def test_runtime_config_snapshot_redacts_secrets_and_is_verifiable() -> None:
    snapshot = build_runtime_config_snapshot(
        {"max_positions": 4, "api_key": "secret", "nested": {"password": "secret"}},
        environment="paper",
        changed_by="operator",
    )
    verify_runtime_config_snapshot(snapshot)
    assert snapshot["values"]["api_key"] == "[REDACTED]"
    assert snapshot["values"]["nested"]["password"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_promotion_and_config_registries_preserve_approval_and_rollback_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from db import session as db_session
    from db.session import close_db, get_session_ctx, init_db

    await close_db()
    monkeypatch.setattr(
        db_session.settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'governance.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            promotions = PromotionRegistry(session)
            review = await promotions.request(
                artifact_id="strategy:v2",
                strategy_id="strategy",
                strategy_version="2",
                from_stage="paper",
                to_stage="live",
                evidence=_evidence(),
            )
            approved = await promotions.approve(
                review.review_id,
                reviewer="risk_owner",
                reason="paper evidence reviewed",
            )
            assert approved.status == "approved"

            configs = RuntimeConfigRegistry(session)
            parent = await configs.register(
                {"max_positions": 3},
                environment="paper",
                changed_by="risk_owner",
            )
            await configs.activate(parent.version_id, activated_by="risk_owner")
            child = await configs.register(
                {"max_positions": 4},
                environment="paper",
                parent_version=parent.version_id,
                changed_by="risk_owner",
            )
            await configs.activate(child.version_id, activated_by="risk_owner")
            restored = await configs.rollback(
                child.version_id,
                rolled_back_by="risk_owner",
                reason="drill",
            )
            assert restored.version_id == parent.version_id
            assert restored.status == "active"
            assert child.status == "rolled_back"
    finally:
        await close_db()
