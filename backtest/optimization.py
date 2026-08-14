"""Resumable, time-ordered parameter search for offline BB experiments."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

import pandas as pd

from backtest.reproducibility import normalize_ohlcv_dataframe
from core.experiment_contracts import build_parameter_set, content_sha256

OPTIMIZATION_CONTRACT_VERSION = "bb.parameter-search.v1"
OBJECTIVE_VERSION = "bb.fee-after-multi-objective.v1"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A finite, auditable parameter dimension."""

    name: str
    kind: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or len(name) > 120:
            raise ValueError("parameter name must be non-empty and <= 120 characters")
        object.__setattr__(self, "name", name)
        kind = str(self.kind or "").strip().lower()
        if kind not in {"int", "float", "choice"}:
            raise ValueError("parameter kind must be int, float, or choice")
        object.__setattr__(self, "kind", kind)
        if kind == "choice":
            if not self.values:
                raise ValueError("choice parameter values must not be empty")
            if any(value is None for value in self.values):
                raise ValueError("choice parameter values must not contain null")
            return
        if self.minimum is None or self.maximum is None or self.step is None:
            raise ValueError("numeric parameter requires minimum, maximum, and step")
        if float(self.maximum) < float(self.minimum) or float(self.step) <= 0:
            raise ValueError("numeric parameter bounds or step are invalid")
        if kind == "int" and any(float(value) != int(float(value)) for value in (self.minimum, self.maximum, self.step)):
            raise ValueError("int parameter bounds and step must be integers")

    def candidates(self) -> tuple[Any, ...]:
        if self.kind == "choice":
            return tuple(self.values)
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        step = float(self.step)
        values: list[Any] = []
        current = minimum
        while current <= maximum + max(abs(step) * 1e-9, 1e-12):
            values.append(int(round(current)) if self.kind == "int" else round(current, 12))
            current += step
        if not values:
            raise ValueError(f"parameter {self.name} produced no candidates")
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    dimensions: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("parameter space must contain at least one dimension")
        names = [item.name for item in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")

    @property
    def fingerprint(self) -> str:
        return content_sha256({"dimensions": [item.to_dict() for item in self.dimensions]})

    def generate(self, *, limit: int, random_seed: int) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("candidate limit must be positive")
        values = [item.candidates() for item in self.dimensions]
        combinations = [
            dict(zip((item.name for item in self.dimensions), combination, strict=True))
            for combination in itertools.product(*values)
        ]
        rng = random.Random(int(random_seed))  # noqa: S311 - reproducible search ordering only
        rng.shuffle(combinations)
        selected = combinations[:limit]
        return [
            {
                "candidate_id": f"candidate_{build_parameter_set(item)['sha256'][:20]}",
                "parameter_set": build_parameter_set(item),
                "values": item,
            }
            for item in selected
        ]

    def neighbors(self, values: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return one-step perturbations for every parameter dimension."""

        baseline = dict(values)
        neighbors: dict[str, dict[str, Any]] = {}
        for dimension in self.dimensions:
            candidates = list(dimension.candidates())
            try:
                index = candidates.index(baseline[dimension.name])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"baseline value is outside parameter space: {dimension.name}"
                ) from exc
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(candidates):
                    continue
                candidate = {**baseline, dimension.name: candidates[neighbor_index]}
                neighbors[_candidate_id(candidate)] = candidate
        return list(neighbors.values())


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_id: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    oos_start: int
    oos_end: int

    def __post_init__(self) -> None:
        if self.train_start < 0 or not (
            self.train_start < self.train_end <= self.validation_start < self.validation_end <= self.oos_start < self.oos_end
        ):
            raise ValueError("walk-forward windows must be chronological and non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_walk_forward_windows(
    row_count: int,
    *,
    train_rows: int,
    validation_rows: int,
    oos_rows: int,
    step_rows: int | None = None,
) -> list[WalkForwardWindow]:
    if min(row_count, train_rows, validation_rows, oos_rows) <= 0:
        raise ValueError("walk-forward row counts must be positive")
    step = int(step_rows or oos_rows)
    if step <= 0:
        raise ValueError("walk-forward step_rows must be positive")
    windows: list[WalkForwardWindow] = []
    start = 0
    index = 0
    while start + train_rows + validation_rows + oos_rows <= row_count:
        train_end = start + train_rows
        validation_end = train_end + validation_rows
        oos_end = validation_end + oos_rows
        windows.append(
            WalkForwardWindow(
                window_id=f"wf_{index:03d}",
                train_start=start,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                oos_start=validation_end,
                oos_end=oos_end,
            )
        )
        start += step
        index += 1
    if not windows:
        raise ValueError("dataset is too short for one complete walk-forward window")
    return windows


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    min_trades: int = 10
    min_profit_factor: float = 1.0
    max_drawdown_pct: float = 25.0
    max_tail_loss_pct: float = 10.0
    min_oos_folds: int = 2
    require_oos_positive: bool = True
    net_profit_weight: float = 1.0
    profit_factor_weight: float = 10.0
    drawdown_penalty_weight: float = 0.5
    tail_loss_penalty_weight: float = 0.5
    turnover_penalty_weight: float = 0.05
    insufficient_sample_penalty: float = 25.0

    def __post_init__(self) -> None:
        if self.min_trades < 0 or self.min_oos_folds <= 0:
            raise ValueError("objective sample thresholds are invalid")
        if self.min_profit_factor < 0 or self.max_drawdown_pct < 0 or self.max_tail_loss_pct < 0:
            raise ValueError("objective risk thresholds must be non-negative")


DEFAULT_OBJECTIVE_CONFIG = ObjectiveConfig()


def score_metrics(
    metrics: Mapping[str, Any],
    *,
    config: ObjectiveConfig = DEFAULT_OBJECTIVE_CONFIG,
    enforce_constraints: bool = True,
) -> dict[str, Any]:
    """Score fee-after evidence without using win rate as a promotion objective."""

    net_profit = _number(metrics.get("net_profit"), metrics.get("fee_adjusted_net_profit"))
    if net_profit is None:
        net_profit = _number(metrics.get("total_return_pct"), 0.0) or 0.0
    profit_factor = _number(metrics.get("profit_factor"))
    drawdown = _number(metrics.get("max_drawdown_pct"), metrics.get("max_drawdown")) or 0.0
    tail_loss = abs(
        _number(metrics.get("tail_loss_pct"), metrics.get("worst_trade_pct")) or 0.0
    )
    turnover = _number(metrics.get("turnover"), metrics.get("turnover_cost")) or 0.0
    trades = int(_number(metrics.get("total_trades"), metrics.get("trade_count")) or 0)
    blockers: list[str] = []
    if enforce_constraints:
        if trades < config.min_trades:
            blockers.append("insufficient_trade_sample")
        if config.require_oos_positive and net_profit <= 0:
            blockers.append("fee_after_net_profit_not_positive")
        if profit_factor is None:
            blockers.append("profit_factor_missing")
        elif profit_factor < config.min_profit_factor:
            blockers.append("profit_factor_below_threshold")
        if drawdown > config.max_drawdown_pct:
            blockers.append("max_drawdown_above_threshold")
        if tail_loss > config.max_tail_loss_pct:
            blockers.append("tail_loss_above_threshold")
    score = (
        config.net_profit_weight * net_profit
        + config.profit_factor_weight * max((profit_factor or 0.0) - 1.0, 0.0)
        - config.drawdown_penalty_weight * drawdown
        - config.tail_loss_penalty_weight * tail_loss
        - config.turnover_penalty_weight * turnover
        - (config.insufficient_sample_penalty if trades < config.min_trades else 0.0)
    )
    return {
        "objective_version": OBJECTIVE_VERSION,
        "score": round(score, 12),
        "eligible": not blockers,
        "blockers": blockers,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown,
        "tail_loss_pct": tail_loss,
        "turnover": turnover,
        "total_trades": trades,
        "constraints_enforced": enforce_constraints,
    }


def parameter_stability_report(
    baseline_score: float,
    perturbation_scores: Mapping[str, float],
    *,
    max_relative_drop: float = 0.5,
) -> dict[str, Any]:
    """Reject cliff-like performance changes after small parameter perturbations."""

    if not isfinite(float(baseline_score)) or max_relative_drop < 0:
        raise ValueError("stability inputs are invalid")
    rows = []
    for candidate_id, raw_score in perturbation_scores.items():
        score = float(raw_score)
        drop = baseline_score - score
        denominator = max(abs(baseline_score), 1e-12)
        relative_drop = drop / denominator if drop > 0 else 0.0
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "score": score,
                "relative_drop": relative_drop,
                "stable": relative_drop <= max_relative_drop,
            }
        )
    status = (
        "insufficient_perturbations"
        if not rows
        else "pass"
        if all(row["stable"] for row in rows)
        else "fail"
    )
    return {
        "status": status,
        "baseline_score": baseline_score,
        "max_relative_drop": max_relative_drop,
        "perturbations": rows,
    }


Evaluator = Callable[[Mapping[str, Any], pd.DataFrame, str], Mapping[str, Any]]


def walk_forward_search(
    frame: pd.DataFrame,
    *,
    parameter_space: ParameterSpace,
    windows: Sequence[WalkForwardWindow],
    evaluator: Evaluator,
    random_seed: int,
    candidate_limit: int,
    top_k: int = 3,
    objective: ObjectiveConfig = DEFAULT_OBJECTIVE_CONFIG,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a resumable train/validation/OOS search and return only research evidence."""

    data = normalize_ohlcv_dataframe(frame)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if any(window.oos_end > len(data) for window in windows):
        raise ValueError("walk-forward window exceeds the normalized dataset")
    candidates = parameter_space.generate(limit=candidate_limit, random_seed=random_seed)
    window_payload = [window.to_dict() for window in windows]
    optimization_id = f"opt_{content_sha256({'space': parameter_space.fingerprint, 'windows': window_payload, 'seed': random_seed, 'limit': candidate_limit})[:24]}"
    state = dict(resume_state or {})
    if state and state.get("optimization_id") != optimization_id:
        raise ValueError("resume state belongs to a different optimization contract")
    state.update(
        {
            "optimization_version": OPTIMIZATION_CONTRACT_VERSION,
            "optimization_id": optimization_id,
            "space_fingerprint": parameter_space.fingerprint,
            "random_seed": int(random_seed),
            "candidate_limit": int(candidate_limit),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "running",
        }
    )
    fold_states = {
        str(item.get("window_id")): dict(item)
        for item in state.get("folds", [])
        if isinstance(item, Mapping)
    }
    for window in windows:
        existing = fold_states.get(window.window_id, {"window_id": window.window_id})
        training = dict(existing.get("training") or {})
        validation = dict(existing.get("validation") or {})
        oos = dict(existing.get("oos") or {})
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id not in training:
                metrics = dict(evaluator(candidate["values"], data.iloc[window.train_start : window.train_end], "train"))
                training[candidate_id] = {
                    "parameters": candidate["values"],
                    "metrics": metrics,
                    "score": score_metrics(metrics, config=objective, enforce_constraints=False),
                }
                _checkpoint(state, fold_states, window, training, validation, oos, checkpoint_writer)
        ranked = sorted(
            training.values(),
            key=lambda item: float(item["score"].get("score", float("-inf"))),
            reverse=True,
        )[:top_k]
        existing["training_selected_parameters"] = [item.get("parameters") for item in ranked]
        for item in ranked:
            candidate_id = _candidate_id(item["parameters"])
            if candidate_id not in validation:
                metrics = dict(evaluator(item["parameters"], data.iloc[window.validation_start : window.validation_end], "validation"))
                validation[candidate_id] = {
                    "metrics": metrics,
                    "score": score_metrics(metrics, config=objective, enforce_constraints=True),
                }
                _checkpoint(state, fold_states, window, training, validation, oos, checkpoint_writer)
        validation_ranked = sorted(
            (
                (candidate_id, evidence)
                for candidate_id, evidence in validation.items()
                if evidence["score"].get("eligible") is True
            ),
            key=lambda item: float(item[1]["score"].get("score", float("-inf"))),
            reverse=True,
        )
        existing["validation_selected_candidate_ids"] = [item[0] for item in validation_ranked]
        candidate_values = {
            _candidate_id(item["parameters"]): item["parameters"] for item in ranked
        }
        for candidate_id, _validation_evidence in validation_ranked:
            if candidate_id not in oos:
                values = candidate_values[candidate_id]
                metrics = dict(
                    evaluator(
                        values,
                        data.iloc[window.oos_start : window.oos_end],
                        "oos",
                    )
                )
                baseline_score = score_metrics(
                    metrics,
                    config=objective,
                    enforce_constraints=True,
                )
                perturbation_evidence: dict[str, Any] = {}
                for neighbor in parameter_space.neighbors(values):
                    neighbor_id = _candidate_id(neighbor)
                    neighbor_metrics = dict(
                        evaluator(
                            neighbor,
                            data.iloc[window.oos_start : window.oos_end],
                            "oos_perturbation",
                        )
                    )
                    perturbation_evidence[neighbor_id] = {
                        "parameters": neighbor,
                        "metrics": neighbor_metrics,
                        "score": score_metrics(
                            neighbor_metrics,
                            config=objective,
                            enforce_constraints=True,
                        ),
                    }
                stability = parameter_stability_report(
                    float(baseline_score["score"]),
                    {
                        neighbor_id: float(evidence["score"]["score"])
                        for neighbor_id, evidence in perturbation_evidence.items()
                    },
                )
                oos[candidate_id] = {
                    "parameters": values,
                    "metrics": metrics,
                    "score": baseline_score,
                    "parameter_stability": stability,
                    "perturbations": perturbation_evidence,
                }
                _checkpoint(state, fold_states, window, training, validation, oos, checkpoint_writer)
        existing.update(
            {
                "window_id": window.window_id,
                "window": window.to_dict(),
                "training": training,
                "validation": validation,
                "oos": oos,
                "status": "complete",
            }
        )
        fold_states[window.window_id] = existing
        state["folds"] = list(fold_states.values())
        state["updated_at"] = datetime.now(UTC).isoformat()
        if checkpoint_writer is not None:
            checkpoint_writer(dict(state))

    state["folds"] = [fold_states[window.window_id] for window in windows]
    state["status"] = "complete"
    state["updated_at"] = datetime.now(UTC).isoformat()
    ranking = _oos_ranking(state["folds"], min_oos_folds=objective.min_oos_folds)
    return {
        "optimization_version": OPTIMIZATION_CONTRACT_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "optimization_id": optimization_id,
        "status": "complete",
        "space_fingerprint": parameter_space.fingerprint,
        "candidate_count": len(candidates),
        "fold_count": len(windows),
        "folds": state["folds"],
        "oos_ranking": ranking,
        "resume_state": state,
        "promotion_authority": "research_only_no_automatic_live_promotion",
    }


def _checkpoint(
    state: dict[str, Any],
    fold_states: dict[str, dict[str, Any]],
    window: WalkForwardWindow,
    training: dict[str, Any],
    validation: dict[str, Any],
    oos: dict[str, Any],
    writer: Callable[[dict[str, Any]], None] | None,
) -> None:
    fold_states[window.window_id] = {
        "window_id": window.window_id,
        "window": window.to_dict(),
        "training": training,
        "validation": validation,
        "oos": oos,
        "status": "running",
    }
    state["folds"] = list(fold_states.values())
    state["updated_at"] = datetime.now(UTC).isoformat()
    if writer is not None:
        writer(dict(state))


def _oos_ranking(
    folds: Sequence[Mapping[str, Any]],
    *,
    min_oos_folds: int,
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold in folds:
        for candidate_id, evidence in dict(fold.get("oos") or {}).items():
            grouped[candidate_id].append(dict(evidence))
    ranking = []
    for candidate_id, evidences in grouped.items():
        scores = [float(item["score"].get("score", 0.0)) for item in evidences]
        sample_complete = len(evidences) >= min_oos_folds
        objective_eligible = all(item["score"].get("eligible") for item in evidences)
        stability_eligible = all(
            dict(item.get("parameter_stability") or {}).get("status") == "pass"
            for item in evidences
        )
        eligible = sample_complete and objective_eligible and stability_eligible
        blockers = {
            blocker
            for item in evidences
            for blocker in item["score"].get("blockers", [])
        }
        if not sample_complete:
            blockers.add("insufficient_oos_folds")
        if not stability_eligible:
            blockers.add("parameter_stability_failed_or_missing")
        ranking.append(
            {
                "candidate_id": candidate_id,
                "oos_fold_count": len(evidences),
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "minimum_score": min(scores) if scores else 0.0,
                "eligible_for_promotion_review": eligible,
                "parameter_stability_passed": stability_eligible,
                "blockers": sorted(blockers),
            }
        )
    return sorted(ranking, key=lambda item: (item["eligible_for_promotion_review"], item["average_score"]), reverse=True)


def _candidate_id(values: Mapping[str, Any]) -> str:
    return f"candidate_{build_parameter_set(values)['sha256'][:20]}"


def _number(*values: Any) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            return number
    return None
