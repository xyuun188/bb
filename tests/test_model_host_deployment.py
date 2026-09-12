from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import model_host_deployment as deployment

TARGET = "bb-phase3-llm-target.service"
CONFLICTS = [
    "bb-phase3-llm-decision.service",
    "bb-phase3-llm-expert.service",
    "bb-phase3-llm-risk-review.service",
]


def _candidate() -> dict:
    return {
        "model_id": "qwen3.8-27b",
        "model_path": "/home/linux/trade_models/qwen3.8-27b-awq",
        "tokenizer_path": "/home/linux/trade_models/qwen3.8-27b-awq",
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "runtime": {
            "engine": "vllm",
            "engine_version": "0.17.1",
            "transformers_version": "5.8.0",
        },
        "config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "weight_files": [{"path": "model.safetensors", "size_bytes": 1, "sha256": "c" * 64}],
        "storage_required_free_gib": 20.0,
        "context_length": 8192,
        "max_concurrency": 1,
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }


def _payload() -> dict:
    return {
        "candidate": _candidate(),
        "target_service": TARGET,
        "conflicting_services": CONFLICTS,
        "start_script": "#!/usr/bin/env bash\nset -euo pipefail\n",
        "unit": "[Service]\nExecStart=/data/BB/scripts/start_target_single_model.sh\n",
        "service_manifest": {
            "topology_profile": "target_single_model",
            "candidate_model_id": "qwen3.8-27b",
            "live_routing_enabled": False,
        },
    }


def _adapter_payload() -> dict:
    return {
        "candidate": _candidate(),
        "adapter_path": "/data/BB/models/finquant_target_27b/versions/test-adapter",
        "target_service": TARGET,
        "conflicting_services": CONFLICTS,
        "start_script": "#!/usr/bin/env bash\nset -euo pipefail\n",
        "unit": "[Service]\nExecStart=/data/BB/scripts/start_target_single_model.sh\n",
        "service_manifest": {
            "topology_profile": "target_single_model",
            "candidate_model_id": "qwen3.8-27b",
            "live_routing_enabled": False,
        },
    }


class FakeHost:
    def __init__(self, *, active: set[str] | None = None, enabled: set[str] | None = None):
        self.active_services = set(active or ())
        self.enabled_services = set(enabled or ())
        self.units: dict[str, bytes] = {}
        self.controls: list[tuple[str, str]] = []
        self.fail_ready = False
        self.fail_ready_ports: set[int] = set()
        self.fail_chat_ready = False
        self.fail_restore = False
        self.ignored_controls: set[tuple[str, str]] = set()

    def active(self, service: str) -> bool:
        return service in self.active_services

    def enabled(self, service: str) -> bool:
        return service in self.enabled_services

    def control(self, action: str, service: str) -> None:
        self.controls.append((action, service))
        if (action, service) in self.ignored_controls:
            return
        if action == "stop":
            self.active_services.discard(service)
        elif action == "start":
            self.active_services.add(service)
        elif action == "disable":
            self.enabled_services.discard(service)
        elif action == "enable":
            self.enabled_services.add(service)

    def reload(self) -> None:
        return None

    def install_unit(self, staged: Path, service: str) -> None:
        self.units[service] = staged.read_bytes()

    def restore_unit(self, backup: Path | None, service: str) -> None:
        if self.fail_restore:
            raise OSError("injected restore failure")
        if backup is None:
            self.units.pop(service, None)
        else:
            self.units[service] = backup.read_bytes()

    def unit_bytes(self, service: str) -> bytes | None:
        return self.units.get(service)

    def verify_runtime(self, _candidate: dict) -> None:
        return None

    def ready(self, _model_id: str, *, port: int = 8000) -> None:
        if self.fail_ready or port in self.fail_ready_ports:
            raise RuntimeError(f"injected readiness failure: {port}")

    def chat_ready(self, _model_id: str, *, port: int = 8000) -> None:
        if self.fail_chat_ready:
            raise RuntimeError(f"injected chat readiness failure: {port}")


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = tmp_path / "bb"
    (root / "runtime").mkdir(parents=True)
    monkeypatch.setattr(deployment, "verify_artifacts", lambda _candidate: None)
    return root


@pytest.fixture
def adapter_prepared(
    monkeypatch: pytest.MonkeyPatch,
    prepared: Path,
) -> Path:
    adapter = prepared / "models/finquant_target_27b/versions/test-adapter"
    adapter.mkdir(parents=True)
    monkeypatch.setattr(deployment, "_adapter_path", lambda _value: adapter)
    return prepared


def test_artifact_preflight_failure_does_not_mutate_services(monkeypatch, prepared: Path) -> None:
    host = FakeHost(active=set(CONFLICTS), enabled=set(CONFLICTS))
    monkeypatch.setattr(
        deployment,
        "verify_artifacts",
        lambda _candidate: (_ for _ in ()).throw(ValueError("bad hash")),
    )

    with pytest.raises(ValueError, match="bad hash"):
        deployment.deploy_target(_payload(), host=host, root=prepared)

    assert host.controls == []
    assert host.units == {}


def test_runtime_preflight_failure_does_not_mutate_services(prepared: Path) -> None:
    host = FakeHost(active=set(CONFLICTS), enabled=set(CONFLICTS))
    host.verify_runtime = lambda _candidate: (_ for _ in ()).throw(RuntimeError("runtime mismatch"))

    with pytest.raises(RuntimeError, match="runtime mismatch"):
        deployment.deploy_target(_payload(), host=host, root=prepared)

    assert host.controls == []


def test_readiness_failure_restores_files_unit_and_service_states(prepared: Path) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={TARGET, CONFLICTS[0]})
    old_unit = b"[Service]\nExecStart=/old\n"
    host.units[TARGET] = old_unit
    host.fail_ready = True

    with pytest.raises(RuntimeError, match="readiness failure"):
        deployment.deploy_target(_payload(), host=host, root=prepared)

    assert host.units[TARGET] == old_unit
    assert host.active_services == {CONFLICTS[0]}
    assert host.enabled_services == {TARGET, CONFLICTS[0]}
    assert not (prepared / "scripts/start_target_single_model.sh").exists()
    evidence = list((prepared / "runtime").glob("model-switch-*/result.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["status"] == "rolled_back"


def test_success_enables_target_and_disables_conflicts(prepared: Path) -> None:
    host = FakeHost(active=set(CONFLICTS), enabled=set(CONFLICTS))

    result = deployment.deploy_target(_payload(), host=host, root=prepared)

    assert result["status"] == "shadow"
    assert result["live_routing_enabled"] is False
    assert host.active_services == {TARGET}
    assert host.enabled_services == {TARGET}
    assert host.units[TARGET].startswith(b"[Service]")
    assert (prepared / "manifests/target_model_candidate.json").is_file()


def test_transformers_target_start_script_does_not_use_vllm():
    candidate = _candidate()
    candidate["runtime"] = {
        "engine": "transformers",
        "engine_version": "5.8.1",
        "transformers_version": "5.8.1",
    }
    script = deployment.target_start_script(candidate)
    assert "/data/BB/scripts/target_transformers_api.py" in script
    assert "vllm.entrypoints" not in script
    assert "VLLM_WORKER" not in script


def test_transformers_target_start_script_loads_verified_adapter():
    candidate = _candidate()
    candidate["runtime"] = {
        "engine": "transformers",
        "engine_version": "5.8.1",
        "transformers_version": "5.8.1",
    }
    script = deployment.target_start_script(
        candidate,
        adapter_path="/data/BB/models/finquant_target_27b/versions/v1",
        base_model_name="qwen3.8-27b-base",
    )
    assert "--adapter-path" in script
    assert "/data/BB/models/finquant_target_27b/versions/v1" in script


def test_rollback_failure_preserves_failure_evidence(prepared: Path) -> None:
    host = FakeHost(active=set(CONFLICTS), enabled=set(CONFLICTS))
    host.fail_ready = True
    host.fail_restore = True

    with pytest.raises(RuntimeError, match="rollback incomplete") as error:
        deployment.deploy_target(_payload(), host=host, root=prepared)

    backup = Path(str(error.value).split("see ", 1)[1])
    evidence = json.loads((backup / "result.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "rollback_failed"
    assert "restore_target_unit:OSError" in evidence["errors"]


def test_target_rollback_detects_state_mismatch_after_successful_control_call(
    prepared: Path,
) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    host.fail_ready = True
    host.ignored_controls.add(("start", CONFLICTS[0]))

    with pytest.raises(RuntimeError, match="rollback incomplete") as error:
        deployment.deploy_target(_payload(), host=host, root=prepared)

    backup = Path(str(error.value).split("see ", 1)[1])
    evidence = json.loads((backup / "result.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "rollback_failed"
    assert f"ServiceActiveStateMismatch:{CONFLICTS[0]}" in evidence["errors"]


def test_adapter_service_names_must_be_distinct() -> None:
    payload = _adapter_payload()
    payload["conflicting_services"] = [TARGET]

    with pytest.raises(ValueError, match="target/conflicting service set is invalid"):
        deployment._validate_adapter_payload(payload)


def test_adapter_switch_starts_target_and_retires_conflicts(
    adapter_prepared: Path,
) -> None:
    host = FakeHost(active=set(CONFLICTS), enabled=set(CONFLICTS))

    result = deployment.deploy_target_adapter(
        _adapter_payload(),
        host=host,
        root=adapter_prepared,
    )

    assert result["status"] == "shadow"
    assert result["live_routing_enabled"] is False
    assert host.active_services == {TARGET}
    assert host.enabled_services == {TARGET}
    assert TARGET in host.units
    assert (adapter_prepared / "scripts/start_target_single_model.sh").is_file()


def test_adapter_target_readiness_failure_restores_everything(
    adapter_prepared: Path,
) -> None:
    start_script = adapter_prepared / "scripts/start_target_single_model.sh"
    start_script.parent.mkdir(parents=True)
    start_script.write_bytes(b"old target\n")
    old_unit = b"[Service]\nExecStart=/old-target\n"
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    host.units[TARGET] = old_unit
    host.fail_ready_ports.add(8000)

    with pytest.raises(RuntimeError, match="readiness failure: 8000"):
        deployment.deploy_target_adapter(
            _adapter_payload(),
            host=host,
            root=adapter_prepared,
        )

    assert start_script.read_bytes() == b"old target\n"
    assert host.units[TARGET] == old_unit
    assert host.active_services == {CONFLICTS[0]}
    assert host.enabled_services == {CONFLICTS[0]}
    evidence = list((adapter_prepared / "runtime").glob("model-adapter-switch-*/result.json"))
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["status"] == "rolled_back"


@pytest.mark.parametrize("failure", ["target_ready", "target_chat"])
def test_adapter_inference_failure_restores_service_state(
    adapter_prepared: Path,
    failure: str,
) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    if failure == "target_ready":
        host.fail_ready_ports.add(8000)
    else:
        host.fail_chat_ready = True

    with pytest.raises(RuntimeError, match="readiness failure"):
        deployment.deploy_target_adapter(
            _adapter_payload(),
            host=host,
            root=adapter_prepared,
        )

    assert host.active_services == {CONFLICTS[0]}
    assert host.enabled_services == {CONFLICTS[0]}


def test_adapter_manual_rollback_restores_original_state(adapter_prepared: Path) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    old_unit = b"[Service]\nExecStart=/old-target\n"
    host.units[TARGET] = old_unit

    deployment_state = deployment.deploy_target_adapter(
        _adapter_payload(),
        host=host,
        root=adapter_prepared,
    )
    result = deployment.rollback_target_adapter(
        deployment_state["backup"],
        host=host,
        root=adapter_prepared,
    )

    assert result["status"] == "rolled_back"
    assert host.active_services == {CONFLICTS[0]}
    assert host.enabled_services == {CONFLICTS[0]}
    assert host.units[TARGET] == old_unit


def test_adapter_rollback_state_mismatch_is_recorded(adapter_prepared: Path) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    host.fail_ready_ports.add(8000)
    host.ignored_controls.add(("start", CONFLICTS[0]))

    with pytest.raises(RuntimeError, match="rollback incomplete") as error:
        deployment.deploy_target_adapter(
            _adapter_payload(),
            host=host,
            root=adapter_prepared,
        )

    backup = Path(str(error.value).split("see ", 1)[1])
    evidence = json.loads((backup / "result.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "rollback_failed"
    assert f"ServiceActiveStateMismatch:{CONFLICTS[0]}" in evidence["errors"]


def test_adapter_manual_rollback_rejects_invalid_service_name_in_manifest(
    adapter_prepared: Path,
) -> None:
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    deployed = deployment.deploy_target_adapter(
        _adapter_payload(), host=host, root=adapter_prepared
    )
    backup = Path(deployed["backup"])
    rollback_path = backup / "rollback.json"
    rollback = json.loads(rollback_path.read_text(encoding="utf-8"))
    rollback["service_states"]["../unsafe.service"] = rollback["service_states"].pop(CONFLICTS[0])
    rollback_path.write_text(json.dumps(rollback), encoding="utf-8")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        deployment.rollback_target_adapter(
            backup, host=host, root=adapter_prepared
        )

    evidence = json.loads(
        (backup / "manual-rollback-result.json").read_text(encoding="utf-8")
    )
    assert "InvalidRollbackServiceName" in evidence["errors"]


def test_adapter_manual_rollback_rejects_unsafe_backup_reference(
    adapter_prepared: Path,
) -> None:
    start_script = adapter_prepared / "scripts/start_target_single_model.sh"
    start_script.parent.mkdir(parents=True)
    start_script.write_bytes(b"old target\n")
    host = FakeHost(active={CONFLICTS[0]}, enabled={CONFLICTS[0]})
    deployed = deployment.deploy_target_adapter(
        _adapter_payload(), host=host, root=adapter_prepared
    )
    backup = Path(deployed["backup"])
    rollback_path = backup / "rollback.json"
    rollback = json.loads(rollback_path.read_text(encoding="utf-8"))
    rollback["files"][0]["backup"] = "../outside"
    rollback_path.write_text(json.dumps(rollback), encoding="utf-8")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        deployment.rollback_target_adapter(
            backup, host=host, root=adapter_prepared
        )

    evidence = json.loads(
        (backup / "manual-rollback-result.json").read_text(encoding="utf-8")
    )
    assert "InvalidRollbackBackup" in evidence["errors"]
