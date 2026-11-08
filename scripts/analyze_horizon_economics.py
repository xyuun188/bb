"""Read-only horizon economics report from an existing development JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.safe_output import safe_print  # noqa: E402
from services.horizon_economics import (  # noqa: E402
    assess_baseline_stability,
    evaluate_horizon_economics,
    primary_horizon_verdict,
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format(report: dict[str, object], indent: int) -> str:
    return json.dumps(report, ensure_ascii=False, indent=indent)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare observed gross directional edge against observed round-trip "
            "cost for each development horizon. Read-only; never submits orders."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("nonoverlapping", "all_opportunities"),
        default="nonoverlapping",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--primary-horizon",
        type=int,
        default=None,
        help="runtime primary horizon to judge; defaults to the project contract",
    )
    args = parser.parse_args()

    payload = _load(args.report)
    research = payload.get("research")
    if not isinstance(research, dict):
        parser.error("report has no research object")
    baselines = research.get("fixed_baselines_by_horizon")
    if not isinstance(baselines, dict):
        parser.error("report has no fixed_baselines_by_horizon")

    report = evaluate_horizon_economics(baselines, view=args.view)
    primary_horizon = args.primary_horizon
    if primary_horizon is None:
        from services.ml_training_contract import (
            PRIMARY_PREDICTION_HORIZON_MINUTES,
        )

        primary_horizon = PRIMARY_PREDICTION_HORIZON_MINUTES
    report["primary_horizon_verdict"] = primary_horizon_verdict(
        baselines,
        primary_horizon_minutes=primary_horizon,
        view=args.view,
    )
    folds = research.get("folds")
    if isinstance(folds, list):
        fold_baselines = []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            # Prefer per-fold fixed-side baselines so stability is measured on
            # the same evidence as the aggregate horizon table. The earlier
            # ridge report stores a single selected-policy outer_evidence map;
            # adapt it to the same horizon/side/view shape instead of silently
            # reporting an empty stability set.
            fold_baselines_value = fold.get("fixed_baselines_by_horizon")
            if isinstance(fold_baselines_value, dict) and fold_baselines_value:
                fold_baselines.append(fold_baselines_value)
                continue
            outer = fold.get("outer_evidence")
            if isinstance(outer, dict):
                nested = outer.get("baselines_by_horizon")
                if isinstance(nested, dict) and nested:
                    fold_baselines.append(nested)
                    continue
                if "aggregate_selected_evidence" in fold:
                    # Selected-policy evidence has no side label. Do not invent
                    # long and short copies: a single mixed result cannot prove
                    # that a particular directional hypothesis is stable.
                    continue
                selection = fold.get("selection")
                policy = selection.get("policy") if isinstance(selection, dict) else None
                if not isinstance(policy, dict):
                    policy = fold.get("selected_policy")
                horizon = policy.get("horizon_minutes") if isinstance(policy, dict) else None
                if (
                    horizon is not None
                    and "long" in outer
                    and "short" in outer
                ):
                    fold_baselines.append(
                        {
                            str(horizon): {
                                side: {
                                    name: outer.get(name)
                                    for name in ("all_opportunities", "nonoverlapping")
                                }
                                for side in ("long", "short")
                            }
                        }
                    )
        report["baseline_stability"] = assess_baseline_stability(
            fold_baselines, view=args.view
        )
    safe_print(_format(report, int(args.json_indent or 0)))


if __name__ == "__main__":
    _main()
