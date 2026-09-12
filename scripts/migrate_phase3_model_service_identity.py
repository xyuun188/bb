"""Deploy the verified single-model Phase 3 target on the model host."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import model_host_deployment  # noqa: E402
from core.model_candidate_manifest import (  # noqa: E402
    ALLOWED_MODEL_ROOTS,
    ModelCandidateManifest,
)
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.model_topology import DEFAULT_MODEL_TOPOLOGY_PROFILE  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from services.phase3_server_migration_audit import FORBIDDEN_LEGACY_SERVICE_NAMES  # noqa: E402

TARGET_SERVICE = "bb-phase3-llm-target.service"
TARGET_START_SCRIPT = "/data/BB/scripts/start_target_single_model.sh"
TARGET_RUNTIME_SCRIPT = ROOT / "scripts" / "target_transformers_api.py"
RETIRED_MODEL_SERVICES = tuple(
    dict.fromkeys(
        (
            "bb-phase3-llm-decision.service",
            "bb-phase3-llm-expert.service",
            "bb-phase3-llm-risk-review.service",
            "qwen3-14b-trade.service",
            "bb-finquant-expert-gateway.service",
            "bb-finquant-expert-alias.service",
            "deepseek-r1-14b-risk.service",
            "deepseek-14b-main.service",
            "deepseek-32b-main.service",
            *FORBIDDEN_LEGACY_SERVICE_NAMES,
        )
    )
)


def target_service_manifest(candidate: ModelCandidateManifest) -> dict[str, object]:
    """Render the only executable local-model service for the target profile."""

    candidate.to_topology()
    return {
        "schema_version": 3,
        "policy_id": "phase3_target_single_model.v1",
        "phase3_root": "/data/BB",
        "topology_profile": "target_single_model",
        "candidate_manifest": "/data/BB/manifests/target_model_candidate.json",
        "candidate_model_id": candidate.model_id,
        "candidate_revision": candidate.revision,
        "shadow_only": True,
        "live_routing_enabled": False,
        "services": [
            {
                "slot": "llm_decision_and_expert_carrier",
                "role": "decision_and_expert_carrier",
                "service_name": TARGET_SERVICE,
                "served_model_name": candidate.model_id,
                "model_dir": candidate.model_path,
                "tokenizer_dir": candidate.tokenizer_path,
                "port": 8000,
                "max_model_len": candidate.context_length,
                "max_num_seqs": candidate.max_concurrency,
                "shadow_only": True,
                "live_routing_enabled": False,
            }
        ],
        "cloud_reviewer_required_for_high_risk_entry": True,
        "can_start_trading": False,
    }


def _unit(*, description: str, exec_start: str) -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description={description}
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        User=linux
        WorkingDirectory=/data/BB
        ExecStart={exec_start}
        Restart=always
        RestartSec=5
        LimitNOFILE=65535

        [Install]
        WantedBy=multi-user.target
        """
    )


def render_target_migration(candidate: ModelCandidateManifest) -> str:
    """Generate the tested, transactional target migration command."""

    candidate.to_topology()
    for path in (candidate.model_path, candidate.tokenizer_path):
        if not any(path.startswith(root) for root in ALLOWED_MODEL_ROOTS):
            roots = ", ".join(ALLOWED_MODEL_ROOTS)
            raise ValueError(f"remote model artifacts must be under an approved root: {roots}")
    payload = {
        "candidate": candidate.to_dict(),
        "service_manifest": target_service_manifest(candidate),
        "target_service": TARGET_SERVICE,
        "conflicting_services": list(RETIRED_MODEL_SERVICES),
        "start_script": model_host_deployment.target_start_script(candidate.to_dict()),
        "runtime_script": TARGET_RUNTIME_SCRIPT.read_text(encoding="utf-8"),
        "unit": _unit(description="BB verified single model", exec_start=TARGET_START_SCRIPT),
    }
    code = inspect.getsource(model_host_deployment)
    return (
        "set -euo pipefail\n"
        "test -x /data/BB/envs/target-inference/bin/python\n"
        "python3 - <<'BB_TARGET_DEPLOY_PY'\n"
        + code
        + "\n"
        + f"print(json.dumps(deploy_target(json.loads({json.dumps(payload)!r}))))\n"
        + "BB_TARGET_DEPLOY_PY\n"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--profile", choices=("target_single_model",), default=DEFAULT_MODEL_TOPOLOGY_PROFILE)
    parser.add_argument("--candidate-manifest", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    candidate = ModelCandidateManifest.load(args.candidate_manifest)
    if not args.apply:
        safe_print(json.dumps(target_service_manifest(candidate), ensure_ascii=False, indent=2))
        return 0

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        safe_print(
            run_remote_text(
                ssh,
                render_target_migration(candidate),
                timeout=720,
                check=True,
                max_output_chars=20_000,
            )
        )
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
