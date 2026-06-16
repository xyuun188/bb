from __future__ import annotations

from pathlib import Path

import pytest

from scripts.deploy_local_ai_tools_service import SERVICE_CODE
from scripts.fix_local_ai_tools_service_path import (
    normalize_remote_python_path,
    render_local_ai_tools_service,
)

ROOT = Path(__file__).resolve().parents[1]


def test_local_ai_tools_generated_service_requires_api_key_or_loopback() -> None:
    assert "LOCAL_AI_TOOLS_API_KEY = os.environ.get" in SERVICE_CODE
    assert "def require_api_key(" in SERVICE_CODE
    assert "dependencies=[Depends(require_api_key)]" in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_API_KEY is required for non-loopback access" in SERVICE_CODE
    assert "Bearer {LOCAL_AI_TOOLS_API_KEY}" in SERVICE_CODE


def test_local_ai_tools_generated_service_disables_local_high_risk_review() -> None:
    assert "openai-compatible-risk-review" not in SERVICE_CODE
    assert "SERVED_REVIEW_MODEL" not in SERVICE_CODE
    assert "MAIN_LLM_BASE" not in SERVICE_CODE
    assert "MAIN_LLM_MODEL" not in SERVICE_CODE
    assert "import httpx" not in SERVICE_CODE
    assert '"data": []' in SERVICE_CODE
    assert "status_code=410" in SERVICE_CODE
    assert "Configure HIGH_RISK_REVIEW_*" in SERVICE_CODE


def test_local_ai_tools_health_returns_service_status_without_trained_bundle() -> None:
    assert (
        'if bundle and isinstance(bundle.get("metadata"), dict):\n'
        '        metadata = bundle["metadata"]\n'
        '    return {\n'
        '        "ok": True,'
    ) in SERVICE_CODE
    assert '"trained_models_available": bool(bundle)' in SERVICE_CODE
    assert '"review_backend": "disabled_use_trading_app_online_model"' in SERVICE_CODE


def test_local_ai_tools_generated_service_does_not_use_wildcard_cors() -> None:
    assert 'allow_origins=["*"]' not in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_CORS_ORIGINS" in SERVICE_CODE
    assert "allow_origins=LOCAL_AI_TOOLS_CORS_ORIGINS" in SERVICE_CODE
    assert "allow_credentials=bool(LOCAL_AI_TOOLS_API_KEY)" in SERVICE_CODE


def test_local_ai_tools_generated_service_uses_trusted_model_artifact_boundary() -> None:
    assert "def _trusted_model_artifact_path(path: Path) -> Path:" in SERVICE_CODE
    assert 'target.suffix != ".joblib"' in SERVICE_CODE
    assert "target.is_relative_to(root)" in SERVICE_CODE
    assert "def load_trusted_joblib_bundle(path: Path) -> dict[str, Any]:" in SERVICE_CODE
    assert "not isinstance(value, dict)" in SERVICE_CODE
    assert (
        "def dump_trusted_joblib_bundle(bundle: dict[str, Any], path: Path) -> Path:"
        in SERVICE_CODE
    )
    assert "tempfile.NamedTemporaryFile" in SERVICE_CODE
    assert "os.replace(tmp_path, target)" in SERVICE_CODE
    assert "joblib.load(BUNDLE_PATH)" not in SERVICE_CODE
    assert "joblib.dump(bundle, BUNDLE_PATH)" not in SERVICE_CODE


def test_local_ai_tools_systemd_uses_env_file_for_secrets() -> None:
    source = (ROOT / "scripts" / "deploy_local_ai_tools_service.py").read_text(encoding="utf-8")

    assert "EnvironmentFile=-/data/trade_ai/local_ai_tools.env" in source
    assert "chmod 600 /data/trade_ai/local_ai_tools.env" in source
    assert "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true" in source
    assert "LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002" in source
    assert "LimitNOFILE=65535" in source
    assert "--timeout-keep-alive 5" in source


def test_local_ai_tools_fix_service_uses_remote_posix_python_path() -> None:
    service = render_local_ai_tools_service("/home/linux/anaconda3/envs/trade_ml/bin/python\n")

    assert "Environment=PATH=/home/linux/anaconda3/envs/trade_ml/bin:" in service
    assert (
        "ExecStart=/home/linux/anaconda3/envs/trade_ml/bin/python -m uvicorn "
        "local_ai_tools_api:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 5"
    ) in service
    assert "LimitNOFILE=65535" in service
    assert "EnvironmentFile=-/data/trade_ai/local_ai_tools.env" in service
    assert "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true" in service
    assert "qwen3-32b-main.service" in service
    assert "\\home\\linux" not in service


def test_normalize_remote_python_path_rejects_unsafe_values() -> None:
    assert (
        normalize_remote_python_path(
            "\n/home/linux/anaconda3/envs/trade_ml/bin/python\n/usr/bin/python3\n"
        )
        == "/home/linux/anaconda3/envs/trade_ml/bin/python"
    )
    with pytest.raises(ValueError, match="absolute"):
        normalize_remote_python_path("python3")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_remote_python_path("/home/linux/bad path/python")
