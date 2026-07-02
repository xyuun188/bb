"""Shared extraction of strategy-generated exit policy for runtime consumers."""

from __future__ import annotations

from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _nonempty_dicts(root: dict[str, Any]) -> list[dict[str, Any]]:
    context = _safe_dict(root.get("strategy_learning_context"))
    learning = _safe_dict(root.get("strategy_learning"))
    runtime = _safe_dict(learning.get("runtime"))
    strategy_mode = _safe_dict(root.get("strategy_mode"))
    mode_learning = _safe_dict(strategy_mode.get("strategy_learning"))
    mode_runtime = _safe_dict(mode_learning.get("runtime"))
    structured = _safe_dict(learning.get("structured_params"))
    mode_structured = _safe_dict(mode_learning.get("structured_params"))
    return [
        root,
        context,
        learning,
        runtime,
        strategy_mode,
        mode_learning,
        mode_runtime,
        structured,
        mode_structured,
    ]


def _first_value(sections: list[dict[str, Any]], key: str) -> Any:
    for section in sections:
        if key not in section:
            continue
        value = section.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and not value:
            continue
        return value
    return None


def _first_dict_value(sections: list[dict[str, Any]], key: str) -> dict[str, Any]:
    value = _first_value(sections, key)
    return _safe_dict(value)


def _winner_hold_strength(
    *,
    winner_hold_extension: str,
    profit_lock_multiplier: float,
    payoff_repair_intensity: float,
    winner_hold_dynamic: dict[str, Any],
    exit_preference: dict[str, Any],
) -> float:
    strength = max(payoff_repair_intensity, 0.0)
    for profile_key in ("training", "reflection"):
        profile = _safe_dict(winner_hold_dynamic.get(profile_key))
        if not profile:
            continue
        imbalance = _safe_float(profile.get("imbalance_score"), 0.0)
        if profile.get("triggered"):
            strength = max(strength, imbalance)
        small_win_ratio = _safe_float(profile.get("small_win_ratio"), 0.0)
        large_loss_ratio = _safe_float(profile.get("large_loss_ratio"), 0.0)
        if small_win_ratio > 0 and large_loss_ratio > 0:
            strength = max(
                strength,
                _clamp((small_win_ratio - large_loss_ratio) * 1.15, 0.0, 1.0),
            )
    winner_mode = str(exit_preference.get("winner_mode") or "")
    if winner_hold_extension == "high" or winner_mode == "let_run":
        strength = max(strength, 0.25)
    if profit_lock_multiplier > 1.0:
        strength = max(strength, _clamp((profit_lock_multiplier - 1.0) / 0.80, 0.0, 1.0))
    return _clamp(strength, 0.0, 1.0)


def _loser_exit_strength(
    *,
    loss_exit_aggressiveness: str,
    exit_preference: dict[str, Any],
) -> float:
    loser_mode = str(exit_preference.get("loser_mode") or "")
    loss_exit_bias = _safe_float(exit_preference.get("loss_exit_bias"), 1.0)
    strength = 0.0
    if loss_exit_aggressiveness == "high":
        strength += 0.30
    elif loss_exit_aggressiveness == "low":
        strength -= 0.18
    if loser_mode == "cut_faster":
        strength += 0.26
    elif loser_mode == "give_room":
        strength -= 0.16
    strength += (loss_exit_bias - 1.0) * 0.60
    return _clamp(strength, -0.35, 0.95)


def exit_strategy_policy_from_context(value: Any) -> dict[str, Any]:
    root = _safe_dict(value)
    sections = _nonempty_dicts(root)
    exit_preference = _first_dict_value(sections, "exit_preference")
    winner_hold_dynamic = _first_dict_value(sections, "winner_hold_dynamic")
    winner_hold_extension = str(_first_value(sections, "winner_hold_extension") or "normal")
    profit_lock_multiplier = _clamp(
        _safe_float(_first_value(sections, "profit_lock_min_usdt_multiplier"), 1.0),
        0.80,
        1.80,
    )
    pullback_lock_enabled = bool(_first_value(sections, "pullback_lock_enabled"))
    payoff_repair_intensity = _clamp(
        _safe_float(_first_value(sections, "payoff_repair_intensity"), 0.0),
        0.0,
        1.0,
    )
    loss_exit_aggressiveness = str(_first_value(sections, "loss_exit_aggressiveness") or "normal")
    winner_hold_strength = _winner_hold_strength(
        winner_hold_extension=winner_hold_extension,
        profit_lock_multiplier=profit_lock_multiplier,
        payoff_repair_intensity=payoff_repair_intensity,
        winner_hold_dynamic=winner_hold_dynamic,
        exit_preference=exit_preference,
    )
    loser_exit_strength = _loser_exit_strength(
        loss_exit_aggressiveness=loss_exit_aggressiveness,
        exit_preference=exit_preference,
    )
    return {
        "winner_hold_extension": winner_hold_extension,
        "profit_lock_min_usdt_multiplier": round(profit_lock_multiplier, 6),
        "pullback_lock_enabled": pullback_lock_enabled,
        "payoff_repair_intensity": round(payoff_repair_intensity, 6),
        "winner_hold_dynamic": winner_hold_dynamic,
        "exit_preference": exit_preference,
        "loss_exit_aggressiveness": loss_exit_aggressiveness,
        "winner_hold_strength": round(winner_hold_strength, 6),
        "loser_exit_strength": round(loser_exit_strength, 6),
        "policy": "shared_strategy_exit_runtime_projection",
    }
