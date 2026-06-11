from __future__ import annotations

from typing import Any

import pytest

from core.remote_ai_service_spec import (
    QWEN3_32B_MAIN_SERVICE,
    RemoteVllmServiceSpec,
    data_disk_guard_command,
    qwen3_main_cleanup_command,
)


def _spec_with(**overrides) -> RemoteVllmServiceSpec:
    values: dict[str, Any] = {
        "model_repo": "Qwen/Qwen3-32B-AWQ",
        "modelscope_model": "Qwen/Qwen3-32B-AWQ",
        "model_dir": "/data/trade_models/Qwen/Qwen3-32B-AWQ",
        "served_model_name": "qwen3-32b-trade",
        "service_name": "qwen3-32b-main.service",
        "description": "Qwen3 32B AWQ vLLM OpenAI API",
        "start_script_name": "start_qwen3_32b_main.sh",
        "download_script_name": "download_qwen3_32b_awq.sh",
        "log_name": "qwen3_32b_main.log",
    }
    values.update(overrides)
    return RemoteVllmServiceSpec(**values)


def test_qwen3_32b_main_service_uses_data_disk_paths() -> None:
    spec = QWEN3_32B_MAIN_SERVICE

    assert spec.model_repo == "Qwen/Qwen3-32B-AWQ"
    assert spec.model_dir == "/data/trade_models/Qwen/Qwen3-32B-AWQ"
    assert spec.start_script_path == "/data/trade_ai/scripts/start_qwen3_32b_main.sh"
    assert spec.staged_service_path == "/data/trade_ai/systemd/qwen3-32b-main.service"
    assert spec.log_path == "/data/trade_ai/logs/qwen3_32b_main.log"


def test_remote_service_spec_rejects_system_disk_paths() -> None:
    with pytest.raises(ValueError, match="/data"):
        _spec_with(model_dir="/root/models/Qwen3-32B-AWQ")


def test_remote_service_spec_rejects_path_traversal_and_unsafe_names() -> None:
    with pytest.raises(ValueError, match="path traversal"):
        _spec_with(model_dir="/data/trade_models/../root")

    with pytest.raises(ValueError, match="simple file name"):
        _spec_with(start_script_name="../start.sh")

    with pytest.raises(ValueError, match="systemd service"):
        _spec_with(service_name="qwen3-32b-main.service;rm")

    with pytest.raises(ValueError, match="control characters"):
        _spec_with(description="Qwen3\nInjected")

    with pytest.raises(ValueError, match="unsupported characters"):
        _spec_with(served_model_name="qwen3 32b")


def test_qwen3_start_script_keeps_vllm_parameters_short_and_awq() -> None:
    script = QWEN3_32B_MAIN_SERVICE.render_start_script()

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "--model /data/trade_models/Qwen/Qwen3-32B-AWQ" in script
    assert "--served-model-name qwen3-32b-trade" in script
    assert "--max-model-len 4096" in script
    assert "--gpu-memory-utilization 0.88" in script
    assert "--quantization awq_marlin" in script
    assert "--max-num-seqs 8" in script
    assert "--max-num-batched-tokens 8192" in script
    assert "--enable-prefix-caching" in script
    assert "--enable-chunked-prefill" in script
    assert "--enforce-eager" not in script
    assert "DeepSeek" not in script
    assert "/root/" not in script


def test_qwen3_systemd_service_points_to_staged_data_disk_script() -> None:
    unit = QWEN3_32B_MAIN_SERVICE.render_systemd_service()

    assert "Description=Qwen3 32B AWQ vLLM OpenAI API" in unit
    assert "WorkingDirectory=/data/trade_ai" in unit
    assert "ExecStart=/data/trade_ai/scripts/start_qwen3_32b_main.sh" in unit
    assert "Restart=always" in unit


def test_qwen3_remote_commands_guard_data_disk_before_start_or_cleanup() -> None:
    guard = data_disk_guard_command()
    install = QWEN3_32B_MAIN_SERVICE.install_and_restart_command()
    cleanup = qwen3_main_cleanup_command()
    temp_path_prefix = "/" + "tmp"

    assert "findmnt -no SOURCE /data" in guard
    assert "exit 2" in guard
    assert install.startswith(guard)
    assert cleanup.startswith(guard)
    assert "deepseek-32b-main.service" in cleanup
    assert "qwen3-32b-main.service" in cleanup
    assert "rm -rf /data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ" in cleanup
    assert temp_path_prefix not in install
    assert temp_path_prefix not in cleanup


def test_qwen3_download_command_uses_quoted_spec_path() -> None:
    command = QWEN3_32B_MAIN_SERVICE.download_and_run_command()

    assert command == (
        "chmod +x /data/trade_ai/scripts/download_qwen3_32b_awq.sh && "
        "/data/trade_ai/scripts/download_qwen3_32b_awq.sh"
    )
    assert ";" not in command
    assert ".." not in command


def test_qwen3_install_command_waits_for_served_model_readiness() -> None:
    readiness = QWEN3_32B_MAIN_SERVICE.readiness_command(attempts=2, sleep_seconds=1)
    install = QWEN3_32B_MAIN_SERVICE.install_and_restart_command()

    assert "http://127.0.0.1:8000/v1/models" in readiness
    assert "grep -F qwen3-32b-trade" in readiness
    assert "vLLM readiness failed for qwen3-32b-trade" in readiness
    assert "systemctl is-active qwen3-32b-main.service && ready=0;" in install
    assert "vLLM model ready: qwen3-32b-trade" in install
