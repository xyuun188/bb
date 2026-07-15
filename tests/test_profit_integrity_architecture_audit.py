from scripts.audit_profit_integrity_architecture import audit, load_deletion_ledger


def test_profit_integrity_architecture_has_no_legacy_production_path() -> None:
    report = audit()

    assert report["status"] == "ok", report["violations"]
    assert report["violations"] == []
    assert report["deletion_ledger_version"] == (
        "2026-07-15.profit-integrity-deletion-ledger.v1"
    )


def test_every_core_semantic_has_one_owner_and_observed_consumers() -> None:
    report = audit()

    for semantic, contract in report["owners"].items():
        assert contract["owner"], semantic
        assert contract["consumer_count"] > 0, semantic


def test_deletion_ledger_has_migration_and_replacement_for_every_path() -> None:
    ledger = load_deletion_ledger()

    assert ledger["optimization_target"] == "maximize_authoritative_fee_after_return_rate"
    for row in ledger["deletions"]:
        assert row["legacy_path"]
        assert row["production_consumers"]
        assert row["test_consumers"]
        assert row["data_migration"]
        assert row["replacement_owner"]
        assert row["deletion_commit"]
