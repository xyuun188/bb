from pathlib import Path

from core.remote_ai_service_spec import QWEN3_8_27B_SERVICE


def test_single_qwen27_service_uses_target_identity_and_safe_resources():
    service = QWEN3_8_27B_SERVICE
    assert service.model_repo == "Qwen/Qwen3.8-27B"
    assert service.model_dir == "/data/BB/models/qwen3.8-27b"
    assert service.served_model_name == "qwen3.8-27b"
    assert service.service_name == "bb-phase3-llm-target.service"
    assert service.port == 8000
    assert service.max_num_seqs == 1
    assert service.gpu_memory_utilization <= 0.80


def test_target_start_script_is_single_carrier_and_bounded():
    script = QWEN3_8_27B_SERVICE.render_start_script()
    assert "--model /data/BB/models/qwen3.8-27b" in script
    assert "--served-model-name qwen3.8-27b" in script
    assert "--max-num-seqs 1" in script
    assert "deepseek" not in script.lower()
    assert "14b" not in script.lower()


def test_target_service_and_readiness_contract_are_consistent():
    unit = QWEN3_8_27B_SERVICE.render_systemd_service()
    readiness = QWEN3_8_27B_SERVICE.readiness_command(attempts=2, sleep_seconds=1)
    install = QWEN3_8_27B_SERVICE.install_and_restart_command()
    assert "ExecStart=/data/trade_ai/scripts/start_qwen3_8_27b.sh" in unit
    assert "grep -F qwen3.8-27b" in readiness
    assert "vLLM model ready: qwen3.8-27b" in readiness
    assert "bb-phase3-llm-target.service" in install


def test_retired_dual_14b_entrypoint_contains_no_runtime_imports():
    assert not Path("scripts/start_dual_14b_llm_tunnel.py").exists()
