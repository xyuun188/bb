from __future__ import annotations

from typing import Any

import pytest

from core.remote_ai_service_spec import (
    DEEPSEEK_R1_14B_RISK_SERVICE,
    QWEN3_14B_TRADE_SERVICE,
    QWEN3_32B_MAIN_SERVICE,
    QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS,
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


def test_dual_14b_services_use_data_disk_and_separate_ports() -> None:
    qwen = QWEN3_14B_TRADE_SERVICE
    r1 = DEEPSEEK_R1_14B_RISK_SERVICE

    assert qwen.model_repo == "Qwen/Qwen3-14B-AWQ"
    assert qwen.model_dir == "/data/trade_models/Qwen/Qwen3-14B-AWQ"
    assert qwen.served_model_name == "qwen3-14b-trade"
    assert qwen.service_name == "qwen3-14b-trade.service"
    assert qwen.port == 8000
    assert qwen.max_model_len == 8192
    assert qwen.gpu_memory_utilization == 0.34
    assert qwen.max_num_seqs == 4
    assert qwen.start_script_path == "/data/trade_ai/scripts/start_qwen3_14b_trade.sh"

    assert r1.model_repo == "casperhansen/deepseek-r1-distill-qwen-14b-awq"
    assert r1.model_dir == "/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq"
    assert r1.served_model_name == "deepseek-r1-14b-risk"
    assert r1.service_name == "deepseek-r1-14b-risk.service"
    assert r1.port == 8003
    assert r1.max_model_len == 4096
    assert r1.gpu_memory_utilization == 0.62
    assert r1.max_num_seqs == 2
    assert r1.start_script_path == "/data/trade_ai/scripts/start_deepseek_r1_14b_risk.sh"


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
    assert "/data/trade_models/Qwen/Qwen3-32B-AWQ" in cleanup
    assert QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS == (
        "/data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ",
        "/data/trade_models/Qwen/Qwen3-32B-AWQ",
    )
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


def test_dual_14b_start_scripts_are_short_context_awq_vllm() -> None:
    qwen_script = QWEN3_14B_TRADE_SERVICE.render_start_script()
    r1_script = DEEPSEEK_R1_14B_RISK_SERVICE.render_start_script()

    assert "--model /data/trade_models/Qwen/Qwen3-14B-AWQ" in qwen_script
    assert "--served-model-name qwen3-14b-trade" in qwen_script
    assert "--port 8000" in qwen_script
    assert "--max-model-len 8192" in qwen_script
    assert "--gpu-memory-utilization 0.34" in qwen_script
    assert "--max-num-seqs 4" in qwen_script
    assert "--quantization awq_marlin" in qwen_script

    assert "--model /data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq" in r1_script
    assert "--served-model-name deepseek-r1-14b-risk" in r1_script
    assert "--port 8003" in r1_script
    assert "--max-model-len 4096" in r1_script
    assert "--gpu-memory-utilization 0.62" in r1_script
    assert "--max-num-batched-tokens 4096" in r1_script
    assert "--max-num-seqs 2" in r1_script
    assert "--quantization awq_marlin" in r1_script


def test_dual_14b_deploy_script_is_plan_first() -> None:
    text = (
        __import__("pathlib")
        .Path("scripts/deploy_dual_14b_llm_services.py")
        .read_text(encoding="utf-8")
    )

    assert "--plan-only" in text
    assert "QWEN3_14B_TRADE_SERVICE" in text
    assert "DEEPSEEK_R1_14B_RISK_SERVICE" in text
    assert "qwen3_main_cleanup_command" in text
