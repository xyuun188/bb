"""Transactional single-model installation on the Linux model host.

Only stdlib imports: the control script sends this exact module to the model
host, so verification and rollback do not depend on its old ML environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

TOKENIZER_NAMES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",
)
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*\.service$")
TARGET_PYTHON = "/data/BB/envs/target-inference/bin/python"
GPU_MEMORY_UTILIZATION = "0.85"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    path = root / relative
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file():
        raise ValueError(f"artifact is not a file: {relative}")
    return path


def tokenizer_fingerprint(root: Path) -> str:
    files = [contained_file(root, name) for name in TOKENIZER_NAMES if (root / name).is_file()]
    if not files:
        raise ValueError("tokenizer files missing")
    payload = "\n".join(f"{item.name}:{hash_file(item)}" for item in files)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_artifacts(candidate: dict) -> None:
    model = Path(candidate["model_path"])
    tokenizer = Path(candidate["tokenizer_path"])
    config_path = contained_file(model, "config.json")
    if hash_file(config_path) != candidate["config_sha256"]:
        raise ValueError("model config fingerprint mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("model_type"):
        raise ValueError("model config has no architecture identity")
    if config.get("model_type") != candidate["model_type"]:
        raise ValueError("model config type differs from manifest")
    if config.get("architectures") != [candidate["architecture"]]:
        raise ValueError("model config architecture differs from manifest")
    if tokenizer_fingerprint(tokenizer) != candidate["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint mismatch")
    declared = {item["path"] for item in candidate["weight_files"]}
    actual = {
        path.relative_to(model).as_posix() for path in model.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    }
    if not declared or actual != declared:
        raise ValueError("weight inventory differs from manifest")
    for item in candidate["weight_files"]:
        path = contained_file(model, item["path"])
        if path.stat().st_size != item["size_bytes"] or hash_file(path) != item["sha256"]:
            raise ValueError(f"weight fingerprint mismatch: {item['path']}")
    index = model / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not weight_map or not set(weight_map.values()).issubset(declared):
            raise ValueError("weight shards referenced by index are missing")
    _verify_available_storage(model, candidate)


def _verify_available_storage(model: Path, candidate: dict) -> None:
    available_gib = shutil.disk_usage(model).free / (1024**3)
    if available_gib < candidate["storage_required_free_gib"]:
        raise ValueError("current model filesystem free space is below the required reserve")


def _safe_service_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not SERVICE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a safe systemd service name")
    return value


def _validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("candidate"), dict):
        raise ValueError("deployment payload must include a candidate object")
    target = _safe_service_name(payload.get("target_service"), field="target_service")
    conflicts = payload.get("conflicting_services")
    if not isinstance(conflicts, list) or not conflicts:
        raise ValueError("deployment payload requires conflicting_services")
    safe_conflicts = [_safe_service_name(item, field="conflicting_services") for item in conflicts]
    if target in safe_conflicts or len(set(safe_conflicts)) != len(safe_conflicts):
        raise ValueError("deployment target/conflicting service set is invalid")
    for field in ("start_script", "unit"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"deployment payload {field} is missing")
    if not isinstance(payload.get("service_manifest"), dict):
        raise ValueError("deployment payload service_manifest is missing")


def atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class LinuxHost:
    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, timeout=60, check=check
        )

    def active(self, service: str) -> bool:
        return self.run("systemctl", "is-active", "--quiet", service, check=False).returncode == 0

    def enabled(self, service: str) -> bool:
        return self.run("systemctl", "is-enabled", "--quiet", service, check=False).returncode == 0

    def control(self, action: str, service: str) -> None:
        self.run("sudo", "-n", "systemctl", action, service)

    def reload(self) -> None:
        self.run("sudo", "-n", "systemctl", "daemon-reload")

    def install_unit(self, staged: Path, service: str) -> None:
        self.run("sudo", "-n", "install", "-m", "0644", str(staged), f"/etc/systemd/system/{service}")

    def restore_unit(self, backup: Path | None, service: str) -> None:
        if backup is not None:
            self.install_unit(backup, service)
        else:
            self.run("sudo", "-n", "rm", "-f", f"/etc/systemd/system/{service}")

    def unit_bytes(self, service: str) -> bytes | None:
        path = Path("/etc/systemd/system") / service
        return path.read_bytes() if path.exists() else None

    def ready(self, model_id: str, timeout: float = 300, *, port: int = 8000) -> None:
        deadline = time.monotonic() + timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                    ids = {row.get("id") for row in json.load(response).get("data", [])}
                if model_id in ids:
                    return
            except (OSError, ValueError, TypeError):
                pass
            time.sleep(2)
        raise RuntimeError("target model identity readiness timed out")

    def chat_ready(self, model_id: str, *, port: int = 8000, timeout: float = 180) -> None:
        payload = json.dumps(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": '/no_think\nReturn exactly this JSON: {"status":"ok"}',
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - fixed loopback URL.
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise RuntimeError("target adapter inference probe failed") from exc
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("target adapter inference response has no choices")

    def verify_runtime(self, candidate: dict) -> None:
        if candidate["runtime"]["engine"] != "vllm":
            raise ValueError("target service currently supports only a verified vLLM runtime")
        probe = (
            "import importlib.metadata as metadata, json, sys; "
            "from transformers import AutoConfig; "
            "config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=False); "
            "print(json.dumps({'vllm': metadata.version('vllm'), "
            "'transformers': metadata.version('transformers'), "
            "'model_type': config.model_type, "
            "'architectures': getattr(config, 'architectures', None)}))"
        )
        result = self.run(TARGET_PYTHON, "-c", probe, candidate["model_path"])
        try:
            measured = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("target runtime probe emitted invalid JSON") from exc
        expected = candidate["runtime"]
        if measured.get("vllm") != expected["engine_version"]:
            raise RuntimeError("target vLLM version differs from the verified candidate")
        if measured.get("transformers") != expected["transformers_version"]:
            raise RuntimeError("target Transformers version differs from the verified candidate")
        if measured.get("model_type") != candidate["model_type"]:
            raise RuntimeError("target runtime resolved a different model type")
        if measured.get("architectures") != [candidate["architecture"]]:
            raise RuntimeError("target runtime resolved a different model architecture")


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # ``msvcrt.locking`` operates from the current file position and is
        # unreliable on append-mode handles.  Keep a single-byte lock region
        # on a read/write handle; closing the handle also releases it if the
        # interpreter is interrupted before the explicit unlock.
        import msvcrt

        lock_path.touch(exist_ok=True)
        with lock_path.open("r+b") as lock:
            lock.seek(0)
            if lock.read(1) != b"0":
                lock.seek(0)
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    # Closing the handle releases the region.  Do not mask a
                    # deployment/rollback exception with a platform cleanup
                    # quirk on the local Windows test host.
                    pass
        return

    import fcntl

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def deploy_target(payload: dict, *, host=None, root: Path = Path("/data/BB")) -> dict:
    """Verify before mutation; restore files and previous service states on failure."""
    host = host or LinuxHost()
    _validate_payload(payload)
    candidate = payload["candidate"]
    verify_artifacts(candidate)
    host.verify_runtime(candidate)
    # This lock serializes model mutations across deploy and maintenance jobs.
    lock_path = root / "runtime/model-maintenance.lock"
    with _exclusive_lock(lock_path):
        return _deploy_locked(payload, host=host, root=root)


def _deploy_locked(payload: dict, *, host, root: Path) -> dict:
    candidate = payload["candidate"]
    service = payload["target_service"]
    conflicts = payload["conflicting_services"]
    states = {name: {"active": host.active(name), "enabled": host.enabled(name)}
              for name in [service, *conflicts]}
    backup = Path(tempfile.mkdtemp(prefix="model-switch-", dir=root / "runtime"))
    old_unit = host.unit_bytes(service)
    if old_unit is not None:
        atomic_write(backup / "target.service", old_unit)
    files = {
        root / "scripts/start_target_single_model.sh": (payload["start_script"].encode(), 0o755),
        root / "manifests/target_model_candidate.json": (json.dumps(candidate).encode(), 0o644),
        root / "manifests/phase3_model_service_manifest.json": (json.dumps(payload["service_manifest"]).encode(), 0o644),
    }
    originals = {path: (path.read_bytes(), path.stat().st_mode & 0o777) if path.exists() else None
                 for path in files}
    for index, (_path, value) in enumerate(originals.items()):
        if value is not None:
            atomic_write(backup / f"file-{index}", value[0], value[1])
    atomic_write(backup / "rollback.json", json.dumps({
        "service_states": states, "paths": [str(path) for path in files],
        "candidate_revision": candidate["revision"], "live_routing_enabled": False,
    }).encode())
    unit_changed = False
    try:
        for name in [service, *conflicts]:
            if states[name]["active"]:
                host.control("stop", name)
            if host.active(name):
                raise RuntimeError(f"conflicting model service did not stop: {name}")
        for path, (content, mode) in files.items():
            atomic_write(path, content, mode)
        staged_unit = backup / "new.service"
        atomic_write(staged_unit, payload["unit"].encode())
        unit_changed = True
        host.install_unit(staged_unit, service)
        host.reload()
        host.control("start", service)
        host.ready(candidate["model_id"])
        host.chat_ready(candidate["model_id"], port=8000)
        if not host.active(service) or any(host.active(name) for name in conflicts):
            raise RuntimeError("target topology service verification failed")
        for name in conflicts:
            if states[name]["enabled"]:
                host.control("disable", name)
        host.control("enable", service)
        expected_states = {
            service: (True, True),
            **{name: (False, False) for name in conflicts},
        }
        for name, (expected_active, expected_enabled) in expected_states.items():
            if host.active(name) != expected_active or host.enabled(name) != expected_enabled:
                raise RuntimeError(f"target service state verification failed: {name}")
    except BaseException as error:
        failures: list[str] = []
        if unit_changed:
            _attempt_restore(
                failures, "stop_target", lambda: host.control("stop", service)
            )
        for index, (path, old) in enumerate(originals.items()):
            if old is None:
                _attempt_restore(
                    failures,
                    f"remove_new_file_{index}",
                    lambda path=path: path.unlink(missing_ok=True),
                )
            else:
                _attempt_restore(
                    failures,
                    f"restore_file_{index}",
                    lambda path=path, old=old: atomic_write(path, old[0], old[1]),
                )
        if unit_changed:
            _attempt_restore(
                failures,
                "restore_target_unit",
                lambda: host.restore_unit(
                    backup / "target.service" if old_unit is not None else None,
                    service,
                ),
            )
            _attempt_restore(failures, "reload_units", host.reload)
        for name, state in states.items():
            if host.enabled(name) != state["enabled"]:
                _attempt_restore(
                    failures,
                    f"restore_enabled_{name}",
                    lambda name=name, state=state: host.control(
                        "enable" if state["enabled"] else "disable", name
                    ),
                )
            if state["active"]:
                _attempt_restore(
                    failures,
                    f"restore_active_{name}",
                    lambda name=name: host.control("start", name),
                )
        failures.extend(_service_state_failures(host, states))
        atomic_write(
            backup / "result.json",
            json.dumps(
                {
                    "status": "rollback_failed" if failures else "rolled_back",
                    "errors": failures,
                    "trigger": type(error).__name__,
                }
            ).encode(),
        )
        if failures:
            raise RuntimeError(f"migration failed and rollback incomplete; see {backup}") from error
        raise
    result = {"status": "shadow", "live_routing_enabled": False, "backup": str(backup),
              "model_id": candidate["model_id"], "revision": candidate["revision"]}
    atomic_write(backup / "result.json", json.dumps(result).encode())
    return result


def _attempt_restore(failures: list[str], label: str, action) -> None:
    """Run one rollback operation and retain a bounded, non-secret error code."""

    try:
        action()
    except Exception as exc:
        failures.append(f"{label}:{type(exc).__name__}")


def _service_state_failures(host, states: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for service, state in states.items():
        expected_active = state.get("active") is True
        expected_enabled = state.get("enabled") is True
        if host.active(service) != expected_active:
            failures.append(f"ServiceActiveStateMismatch:{service}")
        if host.enabled(service) != expected_enabled:
            failures.append(f"ServiceEnabledStateMismatch:{service}")
    return failures


def target_start_script(
    candidate: dict,
    *,
    adapter_path: str | None = None,
    base_model_name: str | None = None,
) -> str:
    served_model_name = candidate["model_id"]
    adapter_args: list[str] = []
    if adapter_path is not None:
        normalized_adapter = str(adapter_path).replace("\\", "/")
        prefix = "/data/BB/models/finquant_target_27b/versions/"
        relative_adapter = normalized_adapter.removeprefix(prefix)
        if (
            not normalized_adapter.startswith(prefix)
            or not relative_adapter
            or any(part in {"", ".", ".."} for part in relative_adapter.split("/"))
        ):
            raise ValueError("target adapter must be an immutable FinQuant 27B version")
        served_model_name = str(base_model_name or "").strip()
        if not served_model_name or served_model_name == candidate["model_id"]:
            raise ValueError("adapter runtime requires a distinct base served-model name")
        adapter_args = [
            "--enable-lora",
            "--lora-modules",
            f"{candidate['model_id']}={adapter_path}",
            "--max-lora-rank",
            "8",
        ]
    args = [
        TARGET_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1", "--port", "8000", "--model", candidate["model_path"],
        "--served-model-name", served_model_name, "--tokenizer", candidate["tokenizer_path"],
        "--max-model-len", str(candidate["context_length"]), "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--max-num-seqs", str(candidate["max_concurrency"]), "--tensor-parallel-size", "1",
        *adapter_args,
    ]
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ -f /data/BB/env/phase3.env ]]; then source /data/BB/env/phase3.env; fi\n"
        "export CUDA_VISIBLE_DEVICES=0\n"
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "exec " + shlex.join(args) + "\n"
    )


def _adapter_path(value: object) -> Path:
    text = str(value or "").replace("\\", "/")
    prefix = "/data/BB/models/finquant_target_27b/versions/"
    relative = text.removeprefix(prefix)
    if (
        not text.startswith(prefix)
        or not relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("adapter path is outside the immutable FinQuant 27B registry")
    return Path(text)


def _validate_adapter_payload(payload: dict) -> None:
    _validate_payload(
        {
            "candidate": payload.get("candidate"),
            "target_service": payload.get("target_service"),
            "conflicting_services": payload.get("conflicting_services"),
            "start_script": payload.get("start_script"),
            "unit": payload.get("unit"),
            "service_manifest": payload.get("service_manifest"),
        }
    )
    _adapter_path(payload.get("adapter_path"))


def deploy_target_adapter(
    payload: dict,
    *,
    host=None,
    root: Path = Path("/data/BB"),
) -> dict:
    """Install one verified LoRA runtime on the canonical target service."""

    host = host or LinuxHost()
    _validate_adapter_payload(payload)
    candidate = payload["candidate"]
    verify_artifacts(candidate)
    host.verify_runtime(candidate)
    adapter = _adapter_path(payload["adapter_path"])
    if not adapter.is_dir():
        raise ValueError("verified target adapter directory does not exist")
    with _exclusive_lock(root / "runtime/model-maintenance.lock"):
        return _deploy_target_adapter_locked(payload, host=host, root=root)


def _deploy_target_adapter_locked(payload: dict, *, host, root: Path) -> dict:
    candidate = payload["candidate"]
    target_service = payload["target_service"]
    services = (target_service, *payload["conflicting_services"])
    states = {
        name: {"active": host.active(name), "enabled": host.enabled(name)}
        for name in services
    }
    backup = Path(tempfile.mkdtemp(prefix="model-adapter-switch-", dir=root / "runtime"))
    old_unit = host.unit_bytes(target_service)
    files = {
        root / "scripts/start_target_single_model.sh": (
            payload["start_script"].encode(),
            0o755,
        ),
    }
    originals = {
        path: (path.read_bytes(), path.stat().st_mode & 0o777) if path.exists() else None
        for path in files
    }
    rollback_payload = {
        "service_states": states,
        "files": [
            {
                "path": str(path),
                "present": old is not None,
                "backup": f"file-{index}" if old is not None else None,
                "mode": old[1] if old is not None else None,
            }
            for index, (path, old) in enumerate(originals.items())
        ],
        "target_model_id": candidate["model_id"],
        "live_routing_enabled": False,
        "unit_present": old_unit is not None,
        "unit_backup": "target.service" if old_unit is not None else None,
    }
    for index, old in enumerate(originals.values()):
        if old is not None:
            atomic_write(backup / f"file-{index}", old[0], old[1])
    if old_unit is not None:
        atomic_write(backup / "target.service", old_unit)
    atomic_write(backup / "rollback.json", json.dumps(rollback_payload).encode())
    try:
        for service in services:
            if host.active(service):
                host.control("stop", service)
        for path, (content, mode) in files.items():
            atomic_write(path, content, mode)
        staged_unit = backup / "new-target.service"
        atomic_write(staged_unit, payload["unit"].encode())
        host.install_unit(staged_unit, target_service)
        host.reload()
        host.control("start", target_service)
        host.ready(candidate["model_id"], port=8000)
        host.chat_ready(candidate["model_id"], port=8000)
        if not host.active(target_service):
            raise RuntimeError("target adapter service did not remain active")
        for service in payload["conflicting_services"]:
            if host.enabled(service):
                host.control("disable", service)
        host.control("enable", target_service)
        expected_states = {
            target_service: (True, True),
            **{name: (False, False) for name in payload["conflicting_services"]},
        }
        for service, (expected_active, expected_enabled) in expected_states.items():
            if host.active(service) != expected_active or host.enabled(service) != expected_enabled:
                raise RuntimeError(f"adapter service state verification failed: {service}")
    except BaseException as error:
        failures = _restore_adapter_backup(backup, host=host, root=root)
        atomic_write(
            backup / "result.json",
            json.dumps(
                {
                    "status": "rollback_failed" if failures else "rolled_back",
                    "errors": failures,
                    "trigger": type(error).__name__,
                }
            ).encode(),
        )
        if failures:
            raise RuntimeError(f"adapter deployment failed and rollback incomplete; see {backup}") from error
        raise
    result = {
        "status": "shadow",
        "live_routing_enabled": False,
        "backup": str(backup),
        "model_id": candidate["model_id"],
        "adapter_path": payload["adapter_path"],
    }
    atomic_write(backup / "result.json", json.dumps(result).encode())
    return result


def _restore_adapter_backup(backup: Path, *, host, root: Path) -> list[str]:
    failures: list[str] = []

    try:
        backup.resolve(strict=True).relative_to((root / "runtime").resolve(strict=True))
        rollback = json.loads((backup / "rollback.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [type(exc).__name__]
    states = rollback.get("service_states")
    files = rollback.get("files")
    unit_present = rollback.get("unit_present")
    unit_backup = rollback.get("unit_backup")
    if (
        not isinstance(states, dict)
        or not isinstance(files, list)
        or not isinstance(unit_present, bool)
        or (unit_present and unit_backup != "target.service")
        or (not unit_present and unit_backup is not None)
    ):
        return ["InvalidRollbackManifest"]
    safe_services: dict[str, dict] = {}
    for name, state in states.items():
        try:
            service = _safe_service_name(name, field="rollback.service")
        except ValueError:
            failures.append("InvalidRollbackServiceName")
            continue
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("active"), bool)
            or not isinstance(state.get("enabled"), bool)
        ):
            failures.append(f"InvalidRollbackServiceState:{service}")
            continue
        safe_services[service] = state
    for service in safe_services:
        if host.active(service):
            _attempt_restore(
                failures,
                f"stop_service_{service}",
                lambda service=service: host.control("stop", service),
            )
    restored_paths: set[Path] = set()
    for item in files:
        if not isinstance(item, dict):
            failures.append("InvalidRollbackFile")
            continue
        path = Path(str(item.get("path") or ""))
        try:
            resolved_path = path.resolve(strict=False)
            resolved_path.relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            failures.append("UnsafeRollbackPath")
            continue
        if resolved_path in restored_paths:
            failures.append("DuplicateRollbackPath")
            continue
        restored_paths.add(resolved_path)
        present = item.get("present")
        if not isinstance(present, bool):
            failures.append("InvalidRollbackFileState")
            continue
        if present:
            backup_name = item.get("backup")
            mode = item.get("mode")
            if (
                not isinstance(backup_name, str)
                or not re.fullmatch(r"file-[0-9]+", backup_name)
                or not isinstance(mode, int)
                or not 0 <= mode <= 0o777
            ):
                failures.append("InvalidRollbackBackup")
                continue
            source = backup / backup_name
            try:
                source.resolve(strict=True).relative_to(backup.resolve(strict=True))
                if not source.is_file():
                    raise ValueError("backup entry is not a file")
            except (OSError, ValueError):
                failures.append("InvalidRollbackBackup")
                continue
            _attempt_restore(
                failures,
                f"restore_file_{backup_name}",
                lambda path=path, source=source, mode=mode: atomic_write(
                    path, source.read_bytes(), mode
                ),
            )
        else:
            _attempt_restore(
                failures,
                "remove_new_file",
                lambda path=path: path.unlink(missing_ok=True),
            )
    target_service = next(
        (service for service in safe_services if service.endswith("-llm-target.service")),
        None,
    )
    if target_service is None:
        failures.append("InvalidRollbackTargetService")
    elif unit_present:
        source = backup / unit_backup
        try:
            source.resolve(strict=True).relative_to(backup.resolve(strict=True))
            if not source.is_file():
                raise ValueError("unit backup is not a file")
        except (OSError, ValueError):
            failures.append("InvalidRollbackUnitBackup")
        else:
            _attempt_restore(
                failures,
                "restore_target_unit",
                lambda: host.restore_unit(source, target_service),
            )
    else:
        _attempt_restore(
            failures,
            "remove_target_unit",
            lambda: host.restore_unit(None, target_service),
        )
    _attempt_restore(failures, "reload_units", host.reload)
    for service, state in safe_services.items():
        enabled = state.get("enabled") is True
        active = state.get("active") is True
        if host.enabled(service) != enabled:
            _attempt_restore(
                failures,
                f"restore_enabled_{service}",
                lambda service=service, enabled=enabled: host.control(
                    "enable" if enabled else "disable", service
                ),
            )
        if active:
            _attempt_restore(
                failures,
                f"restore_active_{service}",
                lambda service=service: host.control("start", service),
            )
    failures.extend(_service_state_failures(host, safe_services))
    return failures


def rollback_target_adapter(
    backup_path: str | Path,
    *,
    host=None,
    root: Path = Path("/data/BB"),
) -> dict:
    host = host or LinuxHost()
    backup = Path(backup_path)
    try:
        backup.resolve(strict=False).relative_to((root / "runtime").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError("adapter rollback backup must be inside the runtime directory") from exc
    with _exclusive_lock(root / "runtime/model-maintenance.lock"):
        failures = _restore_adapter_backup(backup, host=host, root=root)
    result = {
        "status": "rollback_failed" if failures else "rolled_back",
        "errors": failures,
        "backup": str(backup),
    }
    atomic_write(backup / "manual-rollback-result.json", json.dumps(result).encode())
    if failures:
        raise RuntimeError(f"adapter rollback incomplete; see {backup}")
    return result
