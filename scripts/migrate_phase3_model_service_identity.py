"""Migrate the verified one-GPU model runtime to canonical Phase 3 service names."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_candidate_manifest import ModelCandidateManifest  # noqa: E402
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from services.phase3_server_migration_audit import (  # noqa: E402
    FORBIDDEN_LEGACY_SERVICE_NAMES,
    MIGRATION_MANIFEST_PATH,
    PHASE3_RESOURCE_POLICY_ID,
    RESOURCE_RELEASE_MARKER_PATH,
)

QWEN_SERVICE = "bb-phase3-llm-decision.service"
EXPERT_SERVICE = "bb-phase3-llm-expert.service"
RISK_SERVICE = "bb-phase3-llm-risk-review.service"

LEGACY_SERVICES = (
    "qwen3-14b-trade.service",
    "bb-finquant-expert-gateway.service",
    "deepseek-r1-14b-risk.service",
    "local-ai-tools.service",
)
CONTROL_FORBIDDEN_SERVICES = tuple(
    dict.fromkeys((*LEGACY_SERVICES, *FORBIDDEN_LEGACY_SERVICE_NAMES))
)

QWEN_START_SCRIPT = "/data/trade_ai/scripts/start_qwen3_14b_trade.sh"
RISK_START_SCRIPT = "/data/trade_ai/scripts/start_deepseek_r1_14b_risk.sh"
EXPERT_GATEWAY_SCRIPT = "/data/BB/services/finquant_expert_gateway/gateway.py"
SERVICE_MANIFEST = "/data/BB/manifests/phase3_model_service_manifest.json"
TARGET_SERVICE = "bb-phase3-llm-target.service"
TARGET_START_SCRIPT = "/data/BB/scripts/start_target_single_model.sh"


def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def target_service_manifest(candidate: ModelCandidateManifest) -> dict[str, object]:
    """Render the only executable local-model service for the target profile."""

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


def _target_start_script(candidate: ModelCandidateManifest) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        source /data/BB/env/phase3.env
        export CUDA_VISIBLE_DEVICES=0
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        exec /data/BB/envs/phase3-quant/bin/python -m vllm.entrypoints.openai.api_server \\
          --host 127.0.0.1 \\
          --port 8000 \\
          --model {sh(candidate.model_path)} \\
          --served-model-name {sh(candidate.model_id)} \\
          --trust-remote-code \\
          --tokenizer {sh(candidate.tokenizer_path)} \\
          --max-model-len {candidate.context_length} \\
          --gpu-memory-utilization 0.88 \\
          --max-num-seqs {candidate.max_concurrency} \\
          --tensor-parallel-size 1 >> /data/BB/logs/services/target-single-model.log 2>&1
        """
    )


def render_target_migration(candidate: ModelCandidateManifest) -> str:
    """Generate an explicit, fail-closed target migration command."""

    manifest = json.dumps(target_service_manifest(candidate), ensure_ascii=False, indent=2, sort_keys=True)
    manifest_block = textwrap.indent(manifest, "        ")
    candidate_block = textwrap.indent(
        json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), "        "
    )
    start_block = textwrap.indent(_target_start_script(candidate).rstrip(), "        ")
    forbidden_names = " ".join(CONTROL_FORBIDDEN_SERVICES)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        test -x /data/BB/envs/phase3-quant/bin/python
        test -f /data/BB/env/phase3.env
        test -d {sh(candidate.model_path)}
        test -f {sh(candidate.model_path + '/config.json')}
        test -d {sh(candidate.tokenizer_path)}
        mkdir -p /data/BB/scripts /data/BB/logs/services /data/BB/manifests
        cat > /data/BB/manifests/target_model_candidate.json <<'JSON'
{candidate_block}
        JSON
        cat > {TARGET_START_SCRIPT} <<'SCRIPT'
{start_block}
        SCRIPT
        chmod 0755 {TARGET_START_SCRIPT}
        cat > {SERVICE_MANIFEST} <<'JSON'
{manifest_block}
        JSON
        cat > /tmp/{TARGET_SERVICE} <<'UNIT'
        {_unit(description='BB Phase 3 target single verified model', exec_start=TARGET_START_SCRIPT).rstrip()}
        UNIT
        sudo -n install -m 0644 /tmp/{TARGET_SERVICE} /etc/systemd/system/{TARGET_SERVICE}
        sudo -n systemctl disable {forbidden_names} >/dev/null 2>&1 || true
        sudo -n systemctl stop {forbidden_names} >/dev/null 2>&1 || true
        sudo -n systemctl daemon-reload
        sudo -n systemctl enable {TARGET_SERVICE} >/dev/null
        sudo -n systemctl restart {TARGET_SERVICE}
        for i in $(seq 1 120); do
          curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models | grep -F {sh(candidate.model_id)} >/dev/null && break
          sleep 3
        done
        sudo -n systemctl is-active --quiet {TARGET_SERVICE}
        curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models | grep -F {sh(candidate.model_id)} >/dev/null
        for legacy in {forbidden_names}; do
          if systemctl is-active --quiet "$legacy"; then echo "legacy service still active: $legacy" >&2; exit 4; fi
        done
        printf '%s\\n' 'phase3-target-single-model-migrated'
        """
    )


def control_manifests() -> tuple[dict[str, object], dict[str, object]]:
    canonical_services = [QWEN_SERVICE, EXPERT_SERVICE, RISK_SERVICE]
    marker = {
        "schema_version": 1,
        "policy_id": PHASE3_RESOURCE_POLICY_ID,
        "phase3_root": "/data/BB",
        "legacy_resources_stopped": True,
        "old_data_preserved": True,
        "evidence_source": "verified_phase3_service_identity_migration",
        "verification": {
            "canonical_services": canonical_services,
            "legacy_services_stopped": list(CONTROL_FORBIDDEN_SERVICES),
            "verified_model_endpoints": [
                "http://127.0.0.1:8000/v1/models",
                "http://127.0.0.1:8002/v1/models",
                "http://127.0.0.1:8003/v1/models",
                "http://127.0.0.1:8101/health",
            ],
            "service_manifest": SERVICE_MANIFEST,
        },
    }
    migration = {
        "schema_version": 1,
        "policy_id": "phase3_whitelist_only_migration.v1",
        "phase3_root": "/data/BB",
        "whitelist_only": True,
        "whole_disk_copy_allowed": False,
        "old_server_assets_migrated": False,
        "items": [
            {
                "category": "approved_phase3_deploy_manifest",
                "source": "current_repository",
                "path": SERVICE_MANIFEST,
                "purpose": "canonical_model_service_identity",
            }
        ],
    }
    return marker, migration


def _unit(*, description: str, exec_start: str, after: str = "network-online.target") -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description={description}
        After={after}
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


def service_manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "policy_id": "phase3_verified_one_gpu_runtime.v2",
        "phase3_root": "/data/BB",
        "topology_source": "observed_gpu_capacity_and_verified_model_endpoints",
        "shadow_only": True,
        "live_routing_enabled": False,
        "services": [
            {
                "slot": "llm_decision_maker",
                "role": "decision_fallback_and_finquant_carrier",
                "service_name": QWEN_SERVICE,
                "served_model_name": "qwen3-14b-trade",
                "port": 8000,
                "shadow_only": True,
                "live_routing_enabled": False,
            },
            {
                "slot": "llm_expert_pool",
                "role": "verified_finquant_expert",
                "service_name": EXPERT_SERVICE,
                "served_model_name": "BB-FinQuant-Expert-14B",
                "port": 8003,
                "shadow_only": True,
                "live_routing_enabled": False,
            },
            {
                "slot": "llm_high_risk_review",
                "role": "high_risk_review",
                "service_name": RISK_SERVICE,
                "served_model_name": "deepseek-r1-14b-risk",
                "port": 8002,
                "shadow_only": True,
                "live_routing_enabled": False,
            },
        ],
    }


def render_remote_migration() -> str:
    qwen_unit = _unit(
        description="BB Phase 3 Qwen14 decision fallback and FinQuant carrier",
        exec_start=QWEN_START_SCRIPT,
    )
    expert_unit = _unit(
        description="BB Phase 3 verified FinQuant expert gateway",
        exec_start=f"/usr/bin/python3 {EXPERT_GATEWAY_SCRIPT}",
        after=f"network-online.target {QWEN_SERVICE}",
    )
    risk_unit = _unit(
        description="BB Phase 3 DeepSeek high-risk review",
        exec_start=RISK_START_SCRIPT,
    )
    manifest = json.dumps(service_manifest(), ensure_ascii=False, indent=2, sort_keys=True)
    resource_marker, migration_manifest = control_manifests()
    resource_marker_text = json.dumps(
        resource_marker, ensure_ascii=False, indent=2, sort_keys=True
    )
    migration_manifest_text = json.dumps(
        migration_manifest, ensure_ascii=False, indent=2, sort_keys=True
    )
    qwen_unit_block = textwrap.indent(qwen_unit.rstrip(), "        ")
    expert_unit_block = textwrap.indent(expert_unit.rstrip(), "        ")
    risk_unit_block = textwrap.indent(risk_unit.rstrip(), "        ")
    manifest_block = textwrap.indent(manifest, "        ")
    resource_marker_block = textwrap.indent(resource_marker_text, "        ")
    migration_manifest_block = textwrap.indent(migration_manifest_text, "        ")
    legacy_names = " ".join(LEGACY_SERVICES)
    control_forbidden_names = " ".join(CONTROL_FORBIDDEN_SERVICES)
    legacy_paths = " ".join(f"/etc/systemd/system/{name}" for name in LEGACY_SERVICES)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        test -x {QWEN_START_SCRIPT}
        test -x {RISK_START_SCRIPT}
        test -f {EXPERT_GATEWAY_SCRIPT}
        curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models | grep -F 'deepseek-r1-14b-risk' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8101/health | grep -F 'phase3_quant_api' >/dev/null
        cat > /tmp/{QWEN_SERVICE} <<'UNIT'
{qwen_unit_block}
        UNIT
        cat > /tmp/{EXPERT_SERVICE} <<'UNIT'
{expert_unit_block}
        UNIT
        cat > /tmp/{RISK_SERVICE} <<'UNIT'
{risk_unit_block}
        UNIT
        sudo -n install -m 0644 /tmp/{QWEN_SERVICE} /etc/systemd/system/{QWEN_SERVICE}
        sudo -n install -m 0644 /tmp/{EXPERT_SERVICE} /etc/systemd/system/{EXPERT_SERVICE}
        sudo -n install -m 0644 /tmp/{RISK_SERVICE} /etc/systemd/system/{RISK_SERVICE}
        mkdir -p /data/BB/manifests
        cat > {SERVICE_MANIFEST} <<'JSON'
{manifest_block}
        JSON
        sudo -n systemctl disable {legacy_names} >/dev/null 2>&1 || true
        sudo -n systemctl stop {legacy_names} >/dev/null 2>&1 || true
        sudo -n systemctl daemon-reload
        sudo -n systemctl enable {QWEN_SERVICE} {EXPERT_SERVICE} {RISK_SERVICE} >/dev/null
        sudo -n systemctl restart {QWEN_SERVICE} {RISK_SERVICE}
        for i in $(seq 1 120); do
          curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null &&
          curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models | grep -F 'deepseek-r1-14b-risk' >/dev/null && break
          sleep 3
        done
        sudo -n systemctl restart {EXPERT_SERVICE}
        for i in $(seq 1 30); do
          curl -fsS --max-time 5 http://127.0.0.1:8003/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null && break
          sleep 2
        done
        sudo -n systemctl is-active --quiet {QWEN_SERVICE}
        sudo -n systemctl is-active --quiet {EXPERT_SERVICE}
        sudo -n systemctl is-active --quiet {RISK_SERVICE}
        curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models | grep -F 'deepseek-r1-14b-risk' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        sudo -n rm -f {legacy_paths}
        sudo -n systemctl daemon-reload
        for legacy in {control_forbidden_names}; do
          if systemctl is-active --quiet "$legacy"; then
            echo "legacy service still active: $legacy" >&2
            exit 4
          fi
        done
        cat > {RESOURCE_RELEASE_MARKER_PATH} <<'JSON'
{resource_marker_block}
        JSON
        cat > {MIGRATION_MANIFEST_PATH} <<'JSON'
{migration_manifest_block}
        JSON
        python3 -c "import json; json.load(open('{RESOURCE_RELEASE_MARKER_PATH}')); json.load(open('{MIGRATION_MANIFEST_PATH}'))"
        printf '%s\\n' 'phase3-model-service-identity-migrated'
        """
    )


def render_remote_control_manifest_sync() -> str:
    service_text = json.dumps(service_manifest(), ensure_ascii=False, indent=2, sort_keys=True)
    resource_marker, migration_manifest = control_manifests()
    marker_text = json.dumps(resource_marker, ensure_ascii=False, indent=2, sort_keys=True)
    migration_text = json.dumps(
        migration_manifest, ensure_ascii=False, indent=2, sort_keys=True
    )
    service_block = textwrap.indent(service_text, "        ")
    marker_block = textwrap.indent(marker_text, "        ")
    migration_block = textwrap.indent(migration_text, "        ")
    forbidden_names = " ".join(CONTROL_FORBIDDEN_SERVICES)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        test -d /data/BB/models
        test -d /data/BB/cache
        test -d /data/BB/training
        test -d /data/BB/runtime
        test -d /data/BB/logs
        test -d /data/BB/manifests
        test -x {QWEN_START_SCRIPT}
        test -x {RISK_START_SCRIPT}
        test -f {EXPERT_GATEWAY_SCRIPT}
        systemctl is-active --quiet {QWEN_SERVICE}
        systemctl is-active --quiet {EXPERT_SERVICE}
        systemctl is-active --quiet {RISK_SERVICE}
        systemctl is-active --quiet bb-phase3-quant-api.service
        for legacy in {forbidden_names}; do
          if systemctl is-active --quiet "$legacy"; then
            echo "legacy service still active: $legacy" >&2
            exit 4
          fi
        done
        curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models | grep -F 'qwen3-14b-trade' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models | grep -F 'deepseek-r1-14b-risk' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models | grep -F 'BB-FinQuant-Expert-14B' >/dev/null
        curl -fsS --max-time 10 http://127.0.0.1:8101/health | grep -F 'phase3_quant_api' >/dev/null
        cat > {SERVICE_MANIFEST} <<'JSON'
{service_block}
        JSON
        cat > {RESOURCE_RELEASE_MARKER_PATH} <<'JSON'
{marker_block}
        JSON
        cat > {MIGRATION_MANIFEST_PATH} <<'JSON'
{migration_block}
        JSON
        python3 -c "import json; json.load(open('{SERVICE_MANIFEST}')); json.load(open('{RESOURCE_RELEASE_MARKER_PATH}')); json.load(open('{MIGRATION_MANIFEST_PATH}'))"
        printf '%s\\n' 'phase3-model-control-manifests-synchronized'
        """
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--sync-control-manifests", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("target_single_model", "legacy_shadow"),
        default="legacy_shadow",
        help="Use target_single_model for any new deployment; legacy_shadow is audit/rollback only.",
    )
    parser.add_argument("--candidate-manifest", default="", help="Verified model-host candidate manifest.")
    args = parser.parse_args(argv)
    if not args.apply and not args.sync_control_manifests:
        if args.profile == "target_single_model":
            if not args.candidate_manifest:
                raise RuntimeError("target_single_model plan requires --candidate-manifest")
            candidate = ModelCandidateManifest.load(args.candidate_manifest)
            safe_print(json.dumps(target_service_manifest(candidate), ensure_ascii=False, indent=2))
        else:
            safe_print(json.dumps(service_manifest(), ensure_ascii=False, indent=2))
        return 0

    if args.profile == "legacy_shadow":
        if args.apply:
            raise RuntimeError(
                "legacy_shadow deployment is retired; use --profile target_single_model "
                "with a verified candidate manifest. Legacy is read-only audit/rollback."
            )
    if args.profile == "target_single_model" and args.apply:
        if not args.candidate_manifest:
            raise RuntimeError("target_single_model migration requires --candidate-manifest")
        candidate = ModelCandidateManifest.load(args.candidate_manifest)
        command = render_target_migration(candidate)
    elif args.profile == "target_single_model":
        raise RuntimeError("target_single_model control-manifest sync requires a running target service")
    else:
        command = render_remote_control_manifest_sync()

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        safe_print(
            run_remote_text(
                ssh,
                (
                    command
                ),
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
