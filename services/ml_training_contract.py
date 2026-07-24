from typing import Any

DECISION_GROUP_PARTITION_VERSION = "2026-07-24.chronological-purged-holdout.v1"
RANDOM_FOREST_MIN_SAMPLES_LEAF = 8
MIN_TRAINING_SAMPLE_COUNT = RANDOM_FOREST_MIN_SAMPLES_LEAF * 2
MIN_TRAINING_DECISION_GROUP_COUNT = 2


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def decision_group_partition_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["partition_report_missing"]

    errors: list[str] = []
    if value.get("version") != DECISION_GROUP_PARTITION_VERSION:
        errors.append("partition_version_invalid")
    if value.get("ready") is not True:
        errors.append("partition_not_ready")
    train_sample_count = _int_value(value.get("train_sample_count"))
    train_group_count = _int_value(value.get("train_decision_group_count"))
    holdout_sample_count = _int_value(value.get("holdout_sample_count"))
    holdout_group_count = _int_value(value.get("holdout_decision_group_count"))
    purged_sample_count = _int_value(value.get("purged_training_sample_count"))
    if train_sample_count < MIN_TRAINING_SAMPLE_COUNT:
        errors.append("training_sample_count_below_model_minimum")
    if train_group_count < MIN_TRAINING_DECISION_GROUP_COUNT:
        errors.append("training_decision_group_count_below_minimum")
    if holdout_sample_count <= 0:
        errors.append("holdout_sample_count_missing")
    if holdout_group_count <= 0:
        errors.append("holdout_decision_group_count_missing")
    if _int_value(value.get("decision_group_overlap_count")) != 0:
        errors.append("decision_group_overlap_detected")
    if value.get("chronological_label_disjoint") is not True:
        errors.append("chronological_label_overlap_detected")
    if _int_value(value.get("minimum_train_sample_count")) != MIN_TRAINING_SAMPLE_COUNT:
        errors.append("minimum_training_sample_contract_mismatch")
    if (
        _int_value(value.get("minimum_train_decision_group_count"))
        != MIN_TRAINING_DECISION_GROUP_COUNT
    ):
        errors.append("minimum_training_group_contract_mismatch")
    if _int_value(value.get("sample_count")) != (
        train_sample_count + holdout_sample_count + purged_sample_count
    ):
        errors.append("partition_sample_accounting_mismatch")
    return errors
