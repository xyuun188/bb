from scripts.migrate_phase3_model_service_identity import (
    CONTROL_FORBIDDEN_SERVICES,
    EXPERT_SERVICE,
    LEGACY_SERVICES,
    QWEN_SERVICE,
    RISK_SERVICE,
    control_manifests,
    render_remote_control_manifest_sync,
    render_remote_migration,
    service_manifest,
)


def test_service_identity_manifest_matches_verified_runtime() -> None:
    payload = service_manifest()
    services = {row["service_name"]: row for row in payload["services"]}

    assert set(services) == {QWEN_SERVICE, EXPERT_SERVICE, RISK_SERVICE}
    assert services[QWEN_SERVICE]["served_model_name"] == "qwen3-14b-trade"
    assert services[EXPERT_SERVICE]["served_model_name"] == "BB-FinQuant-Expert-14B"
    assert services[RISK_SERVICE]["served_model_name"] == "deepseek-r1-14b-risk"
    assert all(row["live_routing_enabled"] is False for row in services.values())


def test_remote_migration_probes_before_removing_allowlisted_legacy_units() -> None:
    rendered = render_remote_migration()

    remove_at = rendered.index("rm -f ")
    for model_name in (
        "BB-FinQuant-Expert-14B",
        "deepseek-r1-14b-risk",
        "phase3_quant_api",
    ):
        assert rendered.index(model_name) < remove_at
    for service in LEGACY_SERVICES:
        assert f"/etc/systemd/system/{service}" in rendered
    assert "rm -rf" not in rendered


def test_control_manifests_are_truthful_whitelist_only_evidence() -> None:
    marker, migration = control_manifests()

    assert marker["legacy_resources_stopped"] is True
    assert marker["old_data_preserved"] is True
    assert marker["phase3_root"] == "/data/BB"
    verification = marker["verification"]
    assert set(verification["canonical_services"]) == {
        QWEN_SERVICE,
        EXPERT_SERVICE,
        RISK_SERVICE,
    }
    assert set(verification["legacy_services_stopped"]) == set(CONTROL_FORBIDDEN_SERVICES)
    assert migration["whitelist_only"] is True
    assert migration["whole_disk_copy_allowed"] is False
    assert migration["old_server_assets_migrated"] is False
    assert migration["items"] == [
        {
            "category": "approved_phase3_deploy_manifest",
            "source": "current_repository",
            "path": "/data/BB/manifests/phase3_model_service_manifest.json",
            "purpose": "canonical_model_service_identity",
        }
    ]


def test_remote_migration_writes_control_manifests_only_after_runtime_verification() -> None:
    rendered = render_remote_migration()

    final_service_check = rendered.rindex("systemctl is-active --quiet")
    marker_write = rendered.index("phase3_resource_release_manifest.json <<'JSON'")
    whitelist_write = rendered.index("phase3_migration_whitelist.json <<'JSON'")
    success = rendered.index("phase3-model-service-identity-migrated")

    assert final_service_check < marker_write < whitelist_write < success
    assert '"whitelist_only": true' in rendered
    assert '"whole_disk_copy_allowed": false' in rendered


def test_control_manifest_sync_verifies_runtime_without_restarting_services() -> None:
    rendered = render_remote_control_manifest_sync()

    for service in (QWEN_SERVICE, EXPERT_SERVICE, RISK_SERVICE):
        assert f"systemctl is-active --quiet {service}" in rendered
    for service in CONTROL_FORBIDDEN_SERVICES:
        assert service in rendered
    for model_id in (
        "qwen3-14b-trade",
        "BB-FinQuant-Expert-14B",
        "deepseek-r1-14b-risk",
        "phase3_quant_api",
    ):
        assert model_id in rendered
    assert " restart " not in rendered
    assert "systemctl restart" not in rendered
    assert "phase3-model-control-manifests-synchronized" in rendered
