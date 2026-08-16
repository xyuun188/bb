"""Build a local ML candidate from all clean shadow backtests."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ml_signal_service import (
    AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
    AUTO_TRAIN_LEASE_STALE_SECONDS,
    LOCAL_ML_TRAINING_SCHEDULER_ID,
    MODEL_TRAINING_STATE_STORE,
    build_training_frame,
    count_shadow_training_rows,
    load_authoritative_trade_training_samples,
    load_shadow_training_rows,
    shadow_training_quality_report,
    train_from_frame,
)
from services.model_training_state import LOCAL_ML_MODEL_IDS
from services.okx_training_gate import okx_training_refresh_gate
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples


async def run_training(
    *,
    skip_quarantine: bool = False,
    persist_artifact: bool = False,
    confirm_phase3_rebuild: bool = False,
) -> dict[str, object]:
    if persist_artifact and not confirm_phase3_rebuild:
        raise ValueError(
            "persist_artifact requires confirm_phase3_rebuild; run preflight first."
        )
    okx_gate = okx_training_refresh_gate()
    if persist_artifact and not bool(okx_gate.get("allowed")):
        raise ValueError(
            "OKX daily reconciliation blocks local ML artifact persist: "
            f"{okx_gate.get('reason')}"
        )
    quarantine_result: dict[str, object] = {
        "skipped": True,
        "reason": "skip_quarantine flag enabled",
    }
    if not persist_artifact:
        quarantine_result = {
            "skipped": True,
            "reason": "phase3_preflight_no_quarantine_writes",
        }
    elif not skip_quarantine:
        quarantine_result = await quarantine_dirty_shadow_samples()

    rows = await load_shadow_training_rows()
    quality_state = shadow_training_quality_report(rows)
    frame = build_training_frame(rows)
    trade_samples = await load_authoritative_trade_training_samples()
    completed_count = await count_shadow_training_rows()
    metadata = train_from_frame(
        frame,
        completed_sample_count=completed_count,
        training_quality_report=quality_state["quality_report"],
        trade_samples=trade_samples,
        persist_artifact=persist_artifact,
    )
    return {
        "metadata": metadata,
        "training_quarantine": quarantine_result,
        "dry_run": not persist_artifact,
        "preflight_only": not persist_artifact,
        "persist_artifact_requested": persist_artifact,
        "confirm_phase3_rebuild": confirm_phase3_rebuild,
        "okx_daily_reconciliation_gate": okx_gate,
        "frame_sample_count": int(len(frame)),
        "loaded_row_count": int(len(rows)),
        "completed_shadow_sample_count": int(completed_count),
        "authoritative_trade_sample_count": len(trade_samples),
    }


async def _main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Train local ML signal model")
    parser.add_argument("--skip-quarantine", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Deprecated alias for the default preflight mode. "
            "Preflight never quarantines rows or writes model artifacts."
        ),
    )
    parser.add_argument(
        "--persist-artifact",
        action="store_true",
        help="Write a candidate artifact after an explicit Phase 3 rebuild confirmation.",
    )
    parser.add_argument(
        "--confirm-phase3-rebuild",
        action="store_true",
        help="Required together with --persist-artifact to build a governed candidate.",
    )
    args = parser.parse_args()

    result = await run_training(
        skip_quarantine=bool(args.skip_quarantine),
        persist_artifact=bool(args.persist_artifact),
        confirm_phase3_rebuild=bool(args.confirm_phase3_rebuild),
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "training_quarantine": result["training_quarantine"],
                "dry_run": result["dry_run"],
                "frame_sample_count": result["frame_sample_count"],
                "loaded_row_count": result["loaded_row_count"],
                "completed_shadow_sample_count": result["completed_shadow_sample_count"],
                "authoritative_trade_sample_count": result[
                    "authoritative_trade_sample_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def main() -> int:
    lease_attempt = MODEL_TRAINING_STATE_STORE.try_acquire_lease(
        scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
        stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
    )
    if not lease_attempt.acquired or lease_attempt.lease is None:
        print(
            json.dumps(
                {
                    "trained": False,
                    "reason": "local_ml_training_already_running",
                    "lease_reason": lease_attempt.reason,
                    "recovered_stale_lease": lease_attempt.recovered_stale_lease,
                },
                ensure_ascii=False,
            )
        )
        return 3

    lease = lease_attempt.lease
    run_id = lease.run_id
    next_check_at = datetime.now(UTC) + timedelta(
        seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS
    )
    try:
        MODEL_TRAINING_STATE_STORE.heartbeat(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            model_ids=LOCAL_ML_MODEL_IDS,
            interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
        )
        MODEL_TRAINING_STATE_STORE.record_check(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            model_ids=LOCAL_ML_MODEL_IDS,
            run_id=run_id,
            force=True,
        )
        MODEL_TRAINING_STATE_STORE.start_run(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            model_ids=LOCAL_ML_MODEL_IDS,
            run_id=run_id,
            trigger_reason="manual_cli",
            timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
        result = asyncio.run(_main())
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        MODEL_TRAINING_STATE_STORE.finish_check(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            model_ids=LOCAL_ML_MODEL_IDS,
            run_id=run_id,
            result={
                "trained": bool(metadata.get("artifact_persisted")),
                "reason": (
                    "manual_training_completed"
                    if metadata.get("artifact_persisted")
                    else "manual_preflight_completed"
                ),
                "artifact_persisted": bool(metadata.get("artifact_persisted")),
                "completed_shadow_sample_count": result.get(
                    "completed_shadow_sample_count"
                ),
                "completed_trade_sample_count": result.get(
                    "authoritative_trade_sample_count"
                ),
                "last_trained_completed_shadow_sample_count": metadata.get(
                    "last_trained_completed_shadow_sample_count"
                ),
                "last_trained_completed_trade_sample_count": metadata.get(
                    "last_trained_completed_trade_sample_count"
                ),
            },
            next_check_at=next_check_at,
        )
        return 0
    except BaseException as exc:
        MODEL_TRAINING_STATE_STORE.record_exception(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            model_ids=LOCAL_ML_MODEL_IDS,
            run_id=run_id,
            error=f"{type(exc).__name__}: {str(exc)[:900]}",
            next_check_at=next_check_at,
        )
        raise
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
