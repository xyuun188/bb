"""Deploy the Phase 3 quant API to the configured model server."""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import textwrap
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.phase3_quant_api_source import SERVICE_CODE

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
PHASE3_API_PORT = 8101
PHASE3_SERVICE_NAME = "bb-phase3-quant-api.service"
PHASE3_APP_DIR = f"{PHASE3_ROOT}/services/phase3_quant_api"
PHASE3_SYSTEMD_DIR = f"{PHASE3_ROOT}/services/systemd"
PHASE3_LOG_DIR = f"{PHASE3_ROOT}/logs/services"
PHASE3_MODEL_DIR = f"{PHASE3_ROOT}/models/local_ai_tools"
PHASE3_RUNTIME_DIR = f"{PHASE3_ROOT}/runtime/phase3_quant_api"
PHASE3_ENV_FILE = f"{PHASE3_ROOT}/env/phase3.env"
PHASE3_PYTHON_BIN = f"{PHASE3_ROOT}/envs/phase3-quant/bin/python"
PHASE3_POLICY_ID = "phase3_quant_api_shadow_contract_v2_2026_06_27"

def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_phase3_quant_api_service() -> str:
    """Render the Phase 3 quant API systemd unit rooted under /data/BB."""

    env_bin = PurePosixPath(PHASE3_PYTHON_BIN).parent.as_posix()
    return (
        textwrap.dedent(
            f"""
            [Unit]
            Description=BB Phase 3 Quant API - local_ai_tools v2 shadow contracts
            After=network-online.target
            Wants=network-online.target

            [Service]
            Type=simple
            User=root
            WorkingDirectory={PHASE3_APP_DIR}
            Environment=PATH={env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            Environment=BB_PHASE3_ROOT={PHASE3_ROOT}
            Environment=PHASE3_QUANT_API_PORT={PHASE3_API_PORT}
            Environment=LOCAL_AI_TOOLS_MODEL_DIR={PHASE3_MODEL_DIR}
            Environment=LOCAL_AI_TOOLS_SHADOW_MAX_WORKERS=1
            Environment=LOCAL_AI_TOOLS_SPECIALIST_NUM_THREADS=1
            Environment=OMP_NUM_THREADS=1
            Environment=MKL_NUM_THREADS=1
            Environment=OPENBLAS_NUM_THREADS=1
            Environment=NUMEXPR_NUM_THREADS=1
            Environment=TOKENIZERS_PARALLELISM=false
            Environment=LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true
            Environment=LOCAL_AI_TOOLS_ISOLATE_TRAINING_PROCESS=true
            Environment=LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002,http://127.0.0.1:18001
            EnvironmentFile=-{PHASE3_ENV_FILE}
            LimitNOFILE=65535
            ExecStart={PHASE3_PYTHON_BIN} -m uvicorn local_ai_tools_api:app --host 127.0.0.1 --port {PHASE3_API_PORT} --timeout-keep-alive 30
            KillMode=mixed
            TimeoutStopSec=20
            Restart=always
            RestartSec=5
            StandardOutput=append:{PHASE3_LOG_DIR}/phase3_quant_api.log
            StandardError=append:{PHASE3_LOG_DIR}/phase3_quant_api.err.log

            [Install]
            WantedBy=multi-user.target
            """
        ).strip()
        + "\n"
    )


def render_phase3_deploy_plan() -> dict[str, Any]:
    return {
        "policy_id": PHASE3_POLICY_ID,
        "phase3_root": PHASE3_ROOT,
        "service_name": PHASE3_SERVICE_NAME,
        "app_dir": PHASE3_APP_DIR,
        "systemd_dir": PHASE3_SYSTEMD_DIR,
        "log_dir": PHASE3_LOG_DIR,
        "model_dir": PHASE3_MODEL_DIR,
        "runtime_dir": PHASE3_RUNTIME_DIR,
        "env_file": PHASE3_ENV_FILE,
        "python_bin": PHASE3_PYTHON_BIN,
        "port": PHASE3_API_PORT,
        "health_url": f"http://127.0.0.1:{PHASE3_API_PORT}/health",
        "shadow_only": True,
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "legacy_root_used": False,
    }


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    directory = posixpath.dirname(remote_path)
    run_remote_text(ssh, f"mkdir -p {sh(directory)}", timeout=30, check=True)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _remote_preflight_command() -> str:
    return " && ".join(
        [
            f"test -x {sh(PHASE3_PYTHON_BIN)}",
            f"mkdir -p {sh(PHASE3_APP_DIR)} {sh(PHASE3_SYSTEMD_DIR)} {sh(PHASE3_LOG_DIR)} "
            f"{sh(PHASE3_MODEL_DIR)} {sh(PHASE3_RUNTIME_DIR)} {sh(f'{PHASE3_ROOT}/manifests')} "
            f"{sh(PurePosixPath(PHASE3_ENV_FILE).parent.as_posix())}",
            f"touch {sh(PHASE3_ENV_FILE)}",
            f"chmod 600 {sh(PHASE3_ENV_FILE)}",
            f"{PHASE3_PYTHON_BIN} - <<'PY'\n"
            "import fastapi, joblib, numpy, sklearn, uvicorn\n"
            "print('phase3_quant_api_deps_ok')\n"
            "PY",
        ]
    )


def _remote_training_runtime_compatibility_command() -> str:
    """Verify the target Python can create the isolated training executor."""

    return (
        "set -euo pipefail; "
        f"{PHASE3_PYTHON_BIN} - <<'PY'\n"
        "import sys\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "try:\n"
        "    executor = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)\n"
        "    recycle_mode = 'max_tasks_per_child'\n"
        "except TypeError:\n"
        "    if sys.version_info >= (3, 11):\n"
        "        raise\n"
        "    executor = ProcessPoolExecutor(max_workers=1)\n"
        "    recycle_mode = 'manual'\n"
        "executor.shutdown(wait=False, cancel_futures=True)\n"
        "print('phase3_training_executor_compatibility_ok:' + recycle_mode)\n"
        "PY"
    )


def _stop_legacy_8101_holder_command() -> str:
    """Stop the old ad-hoc 8101 inventory API before systemd owns the port."""

    return "\n".join(
        [
            "set -euo pipefail",
            f"new_service={sh(PHASE3_SERVICE_NAME)}",
            f"new_app={sh(PHASE3_APP_DIR + '/local_ai_tools_api.py')}",
            "holders=$(ss -ltnp 'sport = :8101' 2>/dev/null | sed -n 's/.*pid=\\([0-9][0-9]*\\).*/\\1/p' | sort -u || true)",
            "for pid in ${holders}; do",
            '  [ -n "${pid}" ] || continue',
            "  cmdline=$(tr '\\0' ' ' < /proc/${pid}/cmdline 2>/dev/null || true)",
            "  unit=$(systemctl status ${pid} --no-pager 2>/dev/null | sed -n 's/^.*CGroup: \\/system.slice\\/\\([^ ]*\\.service\\).*$/\\1/p' | head -1 || true)",
            '  if printf \'%s\' "${cmdline}" | grep -F "$new_app" >/dev/null; then',
            "    continue",
            "  fi",
            '  if [ -n "${unit}" ] && [ "${unit}" != "$new_service" ]; then',
            '    sudo systemctl stop "${unit}" || true',
            '    sudo systemctl disable "${unit}" || true',
            "  fi",
            '  if kill -0 "${pid}" 2>/dev/null; then',
            '    sudo kill "${pid}" || true',
            "    sleep 2",
            "  fi",
            '  if kill -0 "${pid}" 2>/dev/null; then',
            '    sudo kill -9 "${pid}" || true',
            "  fi",
            "done",
        ]
    )


def _remote_smoke_command() -> str:
    return (
        f"{PHASE3_PYTHON_BIN} - <<'PY'\n"
        "import json\n"
        "import urllib.request\n"
        "\n"
        f"BASE = 'http://127.0.0.1:{PHASE3_API_PORT}'\n"
        f"ENV_FILE = {PHASE3_ENV_FILE!r}\n"
        "\n"
        "def api_key():\n"
        "    try:\n"
        "        for raw_line in open(ENV_FILE, encoding='utf-8'):\n"
        "            line = raw_line.strip()\n"
        "            if line.startswith('LOCAL_AI_TOOLS_API_KEY='):\n"
        "                return line.split('=', 1)[1].strip().strip(chr(34)).strip(chr(39))\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    return ''\n"
        "\n"
        "def read_json(response):\n"
        "    payload = response.read(4 * 1024 * 1024 + 1)\n"
        "    if len(payload) > 4 * 1024 * 1024:\n"
        "        raise RuntimeError('phase3_quant_api_response_exceeds_4mb')\n"
        "    return json.loads(payload.decode('utf-8'))\n"
        "\n"
        "def get(path):\n"
        "    headers = {}\n"
        "    key = api_key()\n"
        "    if key:\n"
        "        headers['Authorization'] = 'Bearer ' + key\n"
        "    request = urllib.request.Request(BASE + path, headers=headers)\n"
        "    with urllib.request.urlopen(request, timeout=8) as response:\n"
        "        return read_json(response)\n"
        "\n"
        "def post(path, payload):\n"
        "    data = json.dumps(payload).encode('utf-8')\n"
        "    headers = {'Content-Type': 'application/json'}\n"
        "    key = api_key()\n"
        "    if key:\n"
        "        headers['Authorization'] = 'Bearer ' + key\n"
        "    request = urllib.request.Request(\n"
        "        BASE + path,\n"
        "        data=data,\n"
        "        headers=headers,\n"
        "        method='POST',\n"
        "    )\n"
        "    with urllib.request.urlopen(request, timeout=8) as response:\n"
        "        return read_json(response)\n"
        "\n"
        "features = {\n"
        "    'current_price': 100.0,\n"
        "    'close': 100.0,\n"
        "    'returns_1': 0.01,\n"
        "    'returns_5': 0.02,\n"
        "    'returns_20': 0.03,\n"
        "    'rsi_14': 55.0,\n"
        "    'volume_ratio': 1.1,\n"
        "    'horizon_minutes': 5,\n"
        "    'news_sentiment_avg': 0.0,\n"
        "    'social_sentiment_avg': 0.0,\n"
        "    'recent_headlines': ['Market liquidity remains stable.'],\n"
        "    'close_sequence': [100.0 + (index * 0.01) for index in range(30)],\n"
        "    'volume_sequence': [1000.0 + index for index in range(30)],\n"
        "}\n"
        "health = get('/health')\n"
        "lifecycle = health.get('artifact_lifecycle')\n"
        "has_artifact = lifecycle in {'shadow', 'canary', 'active'}\n"
        "live = lifecycle == 'active'\n"
        "profit = post('/profit/predict', {'symbol': 'BTC/USDT', 'features': features})\n"
        "sentiment = post('/sentiment/deep/analyze', {'symbol': 'BTC/USDT', 'features': features})\n"
        "timeseries = post('/timeseries/predict', {'symbol': 'BTC/USDT', 'features': features})\n"
        "exit_advice = post('/exit/advise', {'symbol': 'BTC/USDT', 'features': features, 'open_positions': []})\n"
        "assert health.get('service') == 'phase3_quant_api', health\n"
        "assert health.get('root') == '/data/BB', health\n"
        "assert lifecycle in {'unregistered', 'shadow', 'canary', 'active'}, health\n"
        "assert health.get('live_ml_ready') is live, health\n"
        "activation = health.get('artifact_activation_manifest') or {}\n"
        "if has_artifact:\n"
        "    assert activation.get('activation_stage') == lifecycle, health\n"
        "    assert activation.get('live_ml_ready') is live, health\n"
        "    assert activation.get('execution_scope') == ('production' if live else 'paper_only'), health\n"
        "    assert activation.get('paper_execution_permission') is True, health\n"
        "else:\n"
        "    assert not activation, health\n"
        "assert profit.get('trained') is has_artifact, profit\n"
        "assert profit.get('shadow_payload', {}).get('tool') == 'profit_prediction', profit\n"
        "assert profit.get('production_permission') is live, profit\n"
        "assert profit.get('live_ml_ready') is live, profit\n"
        "assert profit.get('prediction_quality', {}).get('production_eligible') is live, profit\n"
        "assert profit.get('prediction_quality', {}).get('paper_eligible') is has_artifact, profit\n"
        "assert profit.get('prediction_quality', {}).get('anomalous') is (not has_artifact), profit\n"
        "assert profit.get('return_distribution_input_version') == '2026-07-15.model-return-distribution-input.v1', profit\n"
        "assert set((profit.get('return_distribution_inputs') or {})) == {'long', 'short'}, profit\n"
        "assert all(item.get('production_eligible') is live for item in (profit.get('return_distribution_inputs') or {}).values()), profit\n"
        "assert all(item.get('paper_eligible') is has_artifact for item in (profit.get('return_distribution_inputs') or {}).values()), profit\n"
        "assert sentiment.get('endpoint') == 'sentiment_deep', sentiment\n"
        "assert sentiment.get('production_permission') is live, sentiment\n"
        "assert timeseries.get('trained') is has_artifact, timeseries\n"
        "assert timeseries.get('production_permission') is live, timeseries\n"
        "assert timeseries.get('live_ml_ready') is live, timeseries\n"
        "if has_artifact:\n"
        "    assert timeseries.get('horizon_minutes') in timeseries.get('available_horizon_minutes', []), timeseries\n"
        "    assert timeseries.get('horizon_selection_policy') == 'best_governed_lower_quantile_native_horizon', timeseries\n"
        "else:\n"
        "    assert timeseries.get('horizon_minutes') == 5, timeseries\n"
        "assert timeseries.get('prediction_quality', {}).get('production_eligible') is live, timeseries\n"
        "assert timeseries.get('prediction_quality', {}).get('paper_eligible') is has_artifact, timeseries\n"
        "assert timeseries.get('prediction_quality', {}).get('anomalous') is (not has_artifact), timeseries\n"
        "assert timeseries.get('return_distribution_input_version') == '2026-07-15.model-return-distribution-input.v1', timeseries\n"
        "assert set((timeseries.get('return_distribution_inputs') or {})) == {'long', 'short'}, timeseries\n"
        "assert all(item.get('production_eligible') is live for item in (timeseries.get('return_distribution_inputs') or {}).values()), timeseries\n"
        "assert all(item.get('paper_eligible') is has_artifact for item in (timeseries.get('return_distribution_inputs') or {}).values()), timeseries\n"
        "if has_artifact:\n"
        "    assert 'loss_probability' in profit, profit\n"
        "else:\n"
        "    assert 'loss_probability' not in profit, profit\n"
        "assert exit_advice.get('action') == 'hold', exit_advice\n"
        "assert exit_advice.get('no_matching_position') is True, exit_advice\n"
        "print(json.dumps({\n"
        "    'event': 'phase3_quant_api_smoke_ok',\n"
        "    'health_contract': {\n"
        "        'artifact_version': health.get('artifact_version'),\n"
        "        'artifact_lifecycle': health.get('artifact_lifecycle'),\n"
        "        'live_ml_ready': health.get('live_ml_ready'),\n"
        "        'training_data_sha256': health.get('training_data_sha256'),\n"
        "        'source_code_sha256': health.get('source_code_sha256'),\n"
        "        'return_evidence_ready': health.get('artifact_activation_manifest', {}).get('return_evidence_ready'),\n"
        "        'return_evidence_blockers': health.get('artifact_activation_manifest', {}).get('return_evidence_blockers'),\n"
        "    },\n"
        "    'profit_contract': {\n"
        "        'shadow_payload': bool(profit.get('shadow_payload')),\n"
        "        'promotion_flow': profit.get('promotion_flow'),\n"
        "        'production_eligible': profit.get('prediction_quality', {}).get('production_eligible'),\n"
        "        'production_permission': profit.get('production_permission'),\n"
        "    },\n"
        "    'sentiment_contract': {\n"
        "        'endpoint': sentiment.get('endpoint'),\n"
        "        'specialist_inference_active': sentiment.get('specialist_inference_active'),\n"
        "        'production_permission': sentiment.get('production_permission'),\n"
        "    },\n"
        "    'timeseries_contract': {\n"
        "        'production_eligible': timeseries.get('prediction_quality', {}).get('production_eligible'),\n"
        "        'paper_eligible': timeseries.get('prediction_quality', {}).get('paper_eligible'),\n"
        "        'production_permission': timeseries.get('production_permission'),\n"
        "    },\n"
        "    'exit_contract': {\n"
        "        'action': exit_advice.get('action'),\n"
        "        'no_matching_position': exit_advice.get('no_matching_position'),\n"
        "    },\n"
        "}, ensure_ascii=False, indent=2, sort_keys=True))\n"
        "PY"
    )


def deploy_phase3_quant_api(*, plan_only: bool = False, start: bool = True) -> None:
    safe_print(
        json.dumps(render_phase3_deploy_plan(), ensure_ascii=False, indent=2, sort_keys=True)
    )
    if plan_only:
        return

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        run_remote_text(ssh, _remote_preflight_command(), timeout=180, check=True)
        _upload_text(ssh, f"{PHASE3_APP_DIR}/local_ai_tools_api.py", SERVICE_CODE)
        run_remote_text(
            ssh,
            _remote_training_runtime_compatibility_command(),
            timeout=60,
            check=True,
        )
        staged_service_path = f"{PHASE3_SYSTEMD_DIR}/{PHASE3_SERVICE_NAME}"
        _upload_text(ssh, staged_service_path, render_phase3_quant_api_service())
        _upload_text(
            ssh,
            f"{PHASE3_ROOT}/manifests/phase3_quant_api_manifest.json",
            json.dumps(render_phase3_deploy_plan(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        run_remote_text(
            ssh,
            f"sudo install -m 0644 {sh(staged_service_path)} /etc/systemd/system/{sh(PHASE3_SERVICE_NAME)} && "
            "sudo systemctl daemon-reload",
            timeout=60,
            check=True,
        )
        if not start:
            safe_print("Phase 3 quant API installed but not started.")
            return
        run_remote_text(
            ssh,
            _stop_legacy_8101_holder_command(),
            timeout=60,
            check=True,
            max_output_chars=20_000,
        )
        run_remote_text(
            ssh,
            f"sudo systemctl enable {sh(PHASE3_SERVICE_NAME)} && "
            f"sudo systemctl restart {sh(PHASE3_SERVICE_NAME)}",
            timeout=90,
            check=True,
        )
        safe_print(
            run_remote_text(
                ssh,
                f"systemctl is-active {sh(PHASE3_SERVICE_NAME)} && "
                "for i in $(seq 1 30); do "
                "  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8101/health/live || true); "
                "  [ \"$code\" = \"200\" ] && break; sleep 2; "
                "done; "
                "[ \"${code:-000}\" = \"200\" ] && "
                + _remote_smoke_command(),
                timeout=180,
                check=True,
                max_output_chars=80_000,
            )
        )
    finally:
        ssh.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Install files and systemd unit without restarting the Phase 3 quant API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deploy_phase3_quant_api(plan_only=bool(args.plan_only), start=not bool(args.install_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
