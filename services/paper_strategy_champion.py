"""Persistent champion/challenger lifecycle for trained-model paper strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_read_session_ctx, get_session_ctx
from models.learning import StrategyProfileSnapshot

PAPER_STRATEGY_CHAMPION_VERSION = "2026-07-21.paper-strategy-champion.v1"
TRAINED_MODEL_STRATEGY_SOURCE = "trained_model_strategy_blueprint"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return default


def _identity(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _conservative_metrics(candidate: dict[str, Any]) -> dict[str, float | None]:
    backtest = _dict(_dict(candidate.get("backtest")).get("metrics"))
    shadow = _dict(_dict(candidate.get("shadow_validation")).get("metrics"))

    def minimum(key: str) -> float | None:
        values = [_float(backtest.get(key)), _float(shadow.get(key))]
        finite = [value for value in values if value is not None]
        return min(finite) if len(finite) == len(values) else None

    drawdowns = [
        _float(backtest.get("max_drawdown")),
        _float(shadow.get("max_drawdown")),
    ]
    return {
        "return_lcb_pct": minimum("return_lcb_pct"),
        "average_net_return_pct": minimum("average_net_return_pct"),
        "profit_factor": minimum("profit_factor"),
        "max_drawdown": (
            max(value for value in drawdowns if value is not None)
            if all(value is not None for value in drawdowns)
            else None
        ),
    }


def build_trained_model_strategy_candidates(
    blueprint: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind validated return partitions to one immutable trained-model blueprint."""

    model_strategy = _dict(blueprint)
    if (
        model_strategy.get("paper_execution_eligible") is not True
        or model_strategy.get("execution_scope") != "paper_only"
        or model_strategy.get("live_execution_permission") is not False
    ):
        return []
    eligible_sides = {
        str(side).lower() for side in model_strategy.get("eligible_sides") or []
    }
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        promotion = _dict(candidate.get("promotion"))
        params = _dict(candidate.get("params"))
        selector = _dict(params.get("selector"))
        provenance = _dict(params.get("policy_provenance"))
        backtest = _dict(candidate.get("backtest"))
        shadow_validation = _dict(candidate.get("shadow_validation"))
        side = str(selector.get("side") or "").lower()
        if (
            promotion.get("production_influence_eligible") is not True
            or side not in eligible_sides
            or provenance.get("evidence_mode")
            != "exact_trained_model_historical_replay"
            or backtest.get("evidence_partition") != "strategy_development"
            or shadow_validation.get("evidence_partition") != "strategy_exam"
            or shadow_validation.get("validation_method")
            != "exact_current_model_on_immutable_shadow_snapshot"
        ):
            continue
        metrics = _conservative_metrics(candidate)
        if any(value is None for value in metrics.values()):
            continue
        historical_id = str(candidate.get("id") or "")
        profile_id = "paper_ml_" + _identity(
            model_strategy.get("strategy_id"),
            historical_id,
        )
        results.append(
            {
                "profile_id": profile_id,
                "version": max(_int(candidate.get("version")), 1),
                "label": f"{candidate.get('label') or historical_id} / trained model",
                "source": TRAINED_MODEL_STRATEGY_SOURCE,
                "description": (
                    "A trained-model return distribution constrained by walk-forward, "
                    "cost-complete shadow validation, and the normal risk pipeline."
                ),
                "model_strategy_id": model_strategy.get("strategy_id"),
                "model_version": model_strategy.get("model_version"),
                "training_data_sha256": model_strategy.get("training_data_sha256"),
                "selector": selector,
                "metrics": metrics,
                "params": {
                    "blueprint": model_strategy,
                    "selector": selector,
                    "historical_return_distribution": _dict(
                        params.get("historical_return_distribution")
                    ),
                },
                "promotion": {
                    "policy_version": PAPER_STRATEGY_CHAMPION_VERSION,
                    "paper_execution_permission": True,
                    "live_execution_permission": False,
                    "can_change_size_or_leverage": False,
                    "can_bypass_order_deduplication": False,
                    "model_quality": _dict(model_strategy.get("model_quality")),
                },
                "backtest_metrics": _dict(candidate.get("backtest")),
                "shadow_validation": _dict(candidate.get("shadow_validation")),
                "rank": _int(candidate.get("rank"), 999_999),
            }
        )
    return sorted(results, key=lambda item: item["rank"])


def compare_paper_strategy_challenger(
    challenger: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    """Require a strict overall improvement without accepting worse strategy risk."""

    challenger_metrics = _dict(challenger.get("metrics"))
    champion_metrics = _dict(champion.get("metrics"))
    missing = [
        key
        for key in (
            "return_lcb_pct",
            "average_net_return_pct",
            "profit_factor",
            "max_drawdown",
        )
        if _float(challenger_metrics.get(key)) is None
        or _float(champion_metrics.get(key)) is None
    ]
    if missing:
        return {
            "accepted": False,
            "reason": "comparison_metric_missing",
            "blocking_reasons": [f"metric_missing:{key}" for key in missing],
        }

    candidate_values = {key: float(challenger_metrics[key]) for key in challenger_metrics}
    champion_values = {key: float(champion_metrics[key]) for key in champion_metrics}
    primary_keys = (
        "return_lcb_pct",
        "average_net_return_pct",
        "profit_factor",
    )
    strategy_strict = all(
        candidate_values[key] > champion_values[key] for key in primary_keys
    )
    strategy_non_degrading = all(
        candidate_values[key] >= champion_values[key] for key in primary_keys
    ) and candidate_values["max_drawdown"] <= champion_values["max_drawdown"]
    different_model = challenger.get("model_version") != champion.get("model_version")
    model_comparison = _dict(
        _dict(challenger.get("promotion")).get("model_quality")
    )
    model_strict = bool(
        different_model
        and model_comparison.get("comparison_accepted") is True
        and model_comparison.get("comparison_reason")
        in {
            "strict_fee_after_improvement",
            "challenger_quality_improved",
            "governed_lifecycle_upgrade",
        }
    )
    accepted = bool(
        strategy_non_degrading and (strategy_strict or model_strict)
    )
    blockers: list[str] = []
    if not strategy_non_degrading:
        blockers.append("challenger_strategy_metrics_or_drawdown_worsened")
    if not strategy_strict and not model_strict:
        blockers.append("challenger_has_no_strict_model_or_strategy_improvement")
    return {
        "accepted": accepted,
        "reason": (
            "strict_strategy_improvement"
            if accepted and strategy_strict
            else "strict_model_improvement_with_non_degrading_strategy"
            if accepted
            else "paper_strategy_champion_retained"
        ),
        "blocking_reasons": blockers,
        "strategy_strict_improvement": strategy_strict,
        "model_strict_improvement": model_strict,
        "challenger_metrics": candidate_values,
        "champion_metrics": champion_values,
    }


def _row_payload(row: StrategyProfileSnapshot) -> dict[str, Any]:
    params = _dict(row.params)
    blueprint = _dict(params.get("blueprint"))
    return {
        "active": bool(row.is_active and not row.is_disabled),
        "profile_id": row.profile_id,
        "profile_version": row.version,
        "status": row.status,
        "label": row.label,
        "source": row.source,
        "execution_scope": "paper_only",
        "paper_execution_permission": bool(
            _dict(row.promotion).get("paper_execution_permission") is True
        ),
        "live_execution_permission": False,
        "model_strategy_id": blueprint.get("strategy_id"),
        "model_version": blueprint.get("model_version"),
        "training_data_sha256": blueprint.get("training_data_sha256"),
        "selector": _dict(params.get("selector")),
        "metrics": _dict(row.backtest_metrics).get("champion_metrics")
        or _dict(row.backtest_metrics).get("metrics")
        or _dict(row.backtest_metrics),
        "promotion": _dict(row.promotion),
        "probe_state": _dict(row.probe_state),
        "scheduler_reason": row.scheduler_reason,
        "policy_version": PAPER_STRATEGY_CHAMPION_VERSION,
    }


class PaperStrategyChampionService:
    """Store one paper champion and keep prior champions available for rollback."""

    async def current(self, mode: str = "paper") -> dict[str, Any]:
        selected_mode = "live" if str(mode).lower() == "live" else "paper"
        if selected_mode == "live":
            return self._inactive("live_strategy_activation_forbidden")
        async with get_read_session_ctx() as session:
            row = await self._active_row(session, lock=False)
            return _row_payload(row) if row is not None else self._inactive(
                "paper_strategy_champion_unavailable"
            )

    async def reconcile(
        self,
        *,
        mode: str,
        blueprint: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        selected_mode = "live" if str(mode).lower() == "live" else "paper"
        if selected_mode == "live":
            return self._inactive("live_strategy_activation_forbidden")
        async with self._session(session) as active_session:
            return await self._reconcile(
                active_session,
                blueprint=_dict(blueprint),
                candidates=candidates,
            )

    @staticmethod
    @asynccontextmanager
    async def _session(session: AsyncSession | None) -> AsyncIterator[AsyncSession]:
        if session is not None:
            yield session
            await session.flush()
            return
        async with get_session_ctx() as owned:
            yield owned

    async def _active_row(
        self,
        session: AsyncSession,
        *,
        lock: bool,
    ) -> StrategyProfileSnapshot | None:
        query = (
            select(StrategyProfileSnapshot)
            .where(
                StrategyProfileSnapshot.execution_mode == "paper",
                StrategyProfileSnapshot.source == TRAINED_MODEL_STRATEGY_SOURCE,
                StrategyProfileSnapshot.is_active.is_(True),
                StrategyProfileSnapshot.is_disabled.is_(False),
            )
            .order_by(StrategyProfileSnapshot.updated_at.desc())
            .limit(1)
        )
        if lock:
            query = query.with_for_update()
        return (await session.execute(query)).scalar_one_or_none()

    async def _reconcile(
        self,
        session: AsyncSession,
        *,
        blueprint: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trained_candidates = build_trained_model_strategy_candidates(
            blueprint,
            candidates,
        )
        rows = [
            await self._persist_candidate(session, candidate)
            for candidate in trained_candidates
        ]
        by_profile = {
            candidate["profile_id"]: (candidate, row)
            for candidate, row in zip(trained_candidates, rows, strict=True)
        }
        champion_row = await self._active_row(session, lock=True)
        if champion_row is None:
            if not trained_candidates:
                return self._inactive("no_validated_trained_model_strategy")
            selected, row = by_profile[trained_candidates[0]["profile_id"]]
            await self._activate(
                session,
                row,
                reason="initial_validated_paper_strategy",
                predecessor=None,
                evidence_version=selected["version"],
            )
            return {
                **_row_payload(row),
                "transition": "initial_champion_activated",
                "challenger_count": len(trained_candidates),
            }

        champion = _row_payload(champion_row)
        current_model_version = str(blueprint.get("model_version") or "")
        champion_model_version = str(champion.get("model_version") or "")
        if current_model_version != champion_model_version:
            return await self._consider_new_model(
                session,
                champion_row=champion_row,
                champion=champion,
                trained_candidates=trained_candidates,
                by_profile=by_profile,
            )

        current_pair = by_profile.get(champion_row.profile_id)
        if current_pair is None:
            return await self._rollback(
                session,
                champion_row,
                reason="active_strategy_no_longer_passes_validation",
            )
        current_candidate, _current_row = current_pair
        evidence_version = _int(current_candidate.get("version"))
        probe = _dict(champion_row.probe_state)
        last_evidence_version = _int(probe.get("last_evaluated_evidence_version"), 0)
        if evidence_version > last_evidence_version:
            health = self._health(current_candidate, champion)
            champion_row.probe_state = {
                **probe,
                "last_evaluated_evidence_version": evidence_version,
                "last_evaluated_at": datetime.now(UTC).isoformat(),
                "last_health": health,
            }
            if health["healthy"] is not True:
                return await self._rollback(
                    session,
                    champion_row,
                    reason="active_strategy_fee_after_performance_degraded",
                )

        for challenger in trained_candidates:
            if challenger["profile_id"] == champion_row.profile_id:
                continue
            comparison = compare_paper_strategy_challenger(challenger, champion)
            if comparison["accepted"] is not True:
                continue
            row = by_profile[challenger["profile_id"]][1]
            await self._activate(
                session,
                row,
                reason=str(comparison["reason"]),
                predecessor=champion_row,
                evidence_version=challenger["version"],
                comparison=comparison,
            )
            return {
                **_row_payload(row),
                "transition": "strictly_better_challenger_activated",
                "comparison": comparison,
                "challenger_count": len(trained_candidates),
            }
        return {
            **_row_payload(champion_row),
            "transition": "champion_retained",
            "challenger_count": len(trained_candidates),
        }

    async def _consider_new_model(
        self,
        session: AsyncSession,
        *,
        champion_row: StrategyProfileSnapshot,
        champion: dict[str, Any],
        trained_candidates: list[dict[str, Any]],
        by_profile: dict[str, tuple[dict[str, Any], StrategyProfileSnapshot]],
    ) -> dict[str, Any]:
        for challenger in trained_candidates:
            comparison = compare_paper_strategy_challenger(challenger, champion)
            if comparison["accepted"] is not True:
                continue
            row = by_profile[challenger["profile_id"]][1]
            await self._activate(
                session,
                row,
                reason=str(comparison["reason"]),
                predecessor=champion_row,
                evidence_version=challenger["version"],
                comparison=comparison,
            )
            return {
                **_row_payload(row),
                "transition": "strictly_better_model_strategy_activated",
                "comparison": comparison,
                "challenger_count": len(trained_candidates),
            }
        return {
            **champion,
            "transition": "model_strategy_rejected_and_rollback_required",
            "model_rollback_required": True,
            "model_rollback_target_version": champion.get("model_version"),
            "challenger_count": len(trained_candidates),
        }

    async def _persist_candidate(
        self,
        session: AsyncSession,
        candidate: dict[str, Any],
    ) -> StrategyProfileSnapshot:
        row = (
            await session.execute(
                select(StrategyProfileSnapshot).where(
                    StrategyProfileSnapshot.execution_mode == "paper",
                    StrategyProfileSnapshot.source == TRAINED_MODEL_STRATEGY_SOURCE,
                    StrategyProfileSnapshot.profile_id == candidate["profile_id"],
                    StrategyProfileSnapshot.version == candidate["version"],
                )
            )
        ).scalar_one_or_none()
        if row is not None:
            return row
        row = StrategyProfileSnapshot(
            execution_mode="paper",
            profile_id=candidate["profile_id"],
            version=candidate["version"],
            label=candidate["label"],
            status="challenger",
            source=candidate["source"],
            description=candidate["description"],
            params=candidate["params"],
            promotion=candidate["promotion"],
            backtest_metrics={
                **candidate["backtest_metrics"],
                "champion_metrics": candidate["metrics"],
            },
            shadow_validation=candidate["shadow_validation"],
            probe_state={
                "last_evaluated_evidence_version": candidate["version"],
            },
            scheduler_reason="awaiting_strict_champion_comparison",
            is_active=False,
            is_disabled=False,
        )
        session.add(row)
        await session.flush()
        return row

    async def _activate(
        self,
        session: AsyncSession,
        row: StrategyProfileSnapshot,
        *,
        reason: str,
        predecessor: StrategyProfileSnapshot | None,
        evidence_version: int,
        comparison: dict[str, Any] | None = None,
        preserve_predecessor: bool = False,
    ) -> None:
        active_rows = list(
            (
                await session.execute(
                    select(StrategyProfileSnapshot).where(
                        StrategyProfileSnapshot.execution_mode == "paper",
                        StrategyProfileSnapshot.source == TRAINED_MODEL_STRATEGY_SOURCE,
                        StrategyProfileSnapshot.is_active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        for active in active_rows:
            if active.id != row.id:
                active.is_active = False
                active.status = "rollback_ready"
                active.scheduler_reason = f"superseded:{reason}"
        promotion = _dict(row.promotion)
        if preserve_predecessor:
            predecessor_profile_id = promotion.get("predecessor_profile_id")
            predecessor_profile_version = promotion.get(
                "predecessor_profile_version"
            )
        else:
            predecessor_profile_id = predecessor.profile_id if predecessor else None
            predecessor_profile_version = predecessor.version if predecessor else None
        row.promotion = {
            **promotion,
            "activation_reason": reason,
            "activated_at": datetime.now(UTC).isoformat(),
            "predecessor_profile_id": predecessor_profile_id,
            "predecessor_profile_version": predecessor_profile_version,
            "comparison": comparison or {},
        }
        row.probe_state = {
            **_dict(row.probe_state),
            "last_evaluated_evidence_version": _int(evidence_version),
            "activated_at": datetime.now(UTC).isoformat(),
        }
        row.status = "champion"
        row.scheduler_reason = reason
        row.is_active = True
        row.is_disabled = False
        await session.flush()

    async def _rollback(
        self,
        session: AsyncSession,
        champion_row: StrategyProfileSnapshot,
        *,
        reason: str,
    ) -> dict[str, Any]:
        promotion = _dict(champion_row.promotion)
        predecessor_id = str(promotion.get("predecessor_profile_id") or "")
        predecessor_version = _int(promotion.get("predecessor_profile_version"), 0)
        predecessor = None
        if predecessor_id and predecessor_version:
            predecessor = (
                await session.execute(
                    select(StrategyProfileSnapshot).where(
                        StrategyProfileSnapshot.execution_mode == "paper",
                        StrategyProfileSnapshot.source == TRAINED_MODEL_STRATEGY_SOURCE,
                        StrategyProfileSnapshot.profile_id == predecessor_id,
                        StrategyProfileSnapshot.version == predecessor_version,
                        StrategyProfileSnapshot.is_disabled.is_(False),
                    )
                )
            ).scalar_one_or_none()
        champion_row.is_active = False
        champion_row.status = "rolled_back"
        champion_row.scheduler_reason = reason
        if predecessor is None:
            await session.flush()
            return {
                **self._inactive("base_strategy_restored_after_strategy_degradation"),
                "transition": "degraded_champion_disabled",
                "rollback_reason": reason,
            }
        await self._activate(
            session,
            predecessor,
            reason=f"automatic_rollback:{reason}",
            predecessor=None,
            evidence_version=_int(
                _dict(predecessor.probe_state).get("last_evaluated_evidence_version")
            ),
            preserve_predecessor=True,
        )
        restored = _row_payload(predecessor)
        return {
            **restored,
            "transition": "previous_champion_restored",
            "rollback_reason": reason,
            "model_rollback_required": (
                restored.get("model_version") != _row_payload(champion_row).get("model_version")
            ),
            "model_rollback_target_version": restored.get("model_version"),
        }

    @staticmethod
    def _health(candidate: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
        current = _dict(candidate.get("metrics"))
        baseline = _dict(champion.get("metrics"))
        reasons: list[str] = []
        for key in (
            "return_lcb_pct",
            "average_net_return_pct",
            "profit_factor",
        ):
            current_value = _float(current.get(key))
            baseline_value = _float(baseline.get(key))
            if current_value is None or baseline_value is None or current_value < baseline_value:
                reasons.append(f"{key}_degraded")
        current_drawdown = _float(current.get("max_drawdown"))
        baseline_drawdown = _float(baseline.get("max_drawdown"))
        if (
            current_drawdown is None
            or baseline_drawdown is None
            or current_drawdown > baseline_drawdown
        ):
            reasons.append("max_drawdown_degraded")
        return {
            "healthy": not reasons,
            "reasons": reasons,
            "current_metrics": current,
            "activation_metrics": baseline,
        }

    @staticmethod
    def _inactive(reason: str) -> dict[str, Any]:
        return {
            "active": False,
            "status": "base_strategy",
            "execution_scope": "paper_only",
            "paper_execution_permission": False,
            "live_execution_permission": False,
            "reason": reason,
            "policy_version": PAPER_STRATEGY_CHAMPION_VERSION,
        }
