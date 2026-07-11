#!/usr/bin/env python3
"""Export BB-FinQuant SFT data and run old-server LoRA specialization.

This script deliberately treats BB-FinQuant-Expert-14B specialization as a real
artifact-producing training job. A renamed Qwen endpoint is not considered
trained unless an adapter directory and specialization manifest are produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import posixpath
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from core.remote_server_info import DEFAULT_ACCOUNT_INFO_DIR, parse_remote_server_info  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_error_text, safe_print  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.learning import ExpertMemory  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _load_authoritative_trade_samples,
    _load_shadow_samples,
    _load_trade_reflection_samples,
    _merge_trade_samples,
)
from services.training_data_quality import annotate_training_payload  # noqa: E402

OLD_PROFILE_FILENAME = "\u5927\u6a21\u578b\u670d\u52a1\u5668\u4fe1\u606f.txt"
REMOTE_ROOT = "/data/BB"
REMOTE_TRAINING_DIR = f"{REMOTE_ROOT}/training/finquant_expert"
REMOTE_SERVICE_DIR = f"{REMOTE_ROOT}/services/finquant_expert_training"
REMOTE_DATASET = f"{REMOTE_TRAINING_DIR}/finquant_sft_latest.jsonl"
REMOTE_DATASET_MANIFEST = f"{REMOTE_TRAINING_DIR}/finquant_sft_latest_manifest.json"
REMOTE_TRAINER = f"{REMOTE_SERVICE_DIR}/train_finquant_lora.py"
REMOTE_ADAPTER_DIR = f"{REMOTE_ROOT}/models/finquant_lora/BB-FinQuant-Expert-14B-v1"
REMOTE_SPECIALIZATION_MANIFEST = f"{REMOTE_ADAPTER_DIR}/specialization_manifest.json"
REMOTE_INFERENCE_BASE_MODEL = "/data/trade_models/Qwen/Qwen3-14B-AWQ"
REMOTE_TRAIN_BASE_REPO = "Qwen/Qwen3-14B"
REMOTE_TRAIN_BASE_MODEL = f"{REMOTE_ROOT}/models/trainable/Qwen3-14B"
REMOTE_QWEN_START_SCRIPT = "/data/trade_ai/scripts/start_qwen3_14b_trade.sh"
REMOTE_ALIAS_SCRIPT = f"{REMOTE_ROOT}/services/model_alias_proxy/finquant_expert_alias.py"
REMOTE_TRAIN_LOG = f"{REMOTE_TRAINING_DIR}/finquant_lora_train.log"
REMOTE_DOWNLOAD_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_download_manifest.json"
REMOTE_VALIDATION_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_validation.json"
REMOTE_PLATFORM_APP_DIR = "/data/bb/app"
REMOTE_PLATFORM_SCRIPT = f"{REMOTE_PLATFORM_APP_DIR}/scripts/finquant_expert_lora_training.py"
REMOTE_PLATFORM_EXPORT_DIR = f"{REMOTE_PLATFORM_APP_DIR}/data/finquant_expert_training"
REMOTE_PLATFORM_EXPORT_WRAPPER = f"{REMOTE_PLATFORM_EXPORT_DIR}/export_wrapper.py"
MODEL_NAME = "BB-FinQuant-Expert-14B"
BASE_MODEL_NAME = "qwen3-14b-trade"
REQUIRED_TRAINING_TABLES = (
    "trade_reflections",
    "positions",
    "orders",
    "shadow_backtests",
    "expert_memories",
)


REMOTE_TRAINER_CODE = r'''
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def row_to_text(tokenizer, row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    parts = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--inference-base-model", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset_path, args.max_samples)
    if not rows:
        raise SystemExit("empty training dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise SystemExit("no trainable LoRA parameters")

    encoded = []
    for row in rows:
        text = row_to_text(tokenizer, row)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_tensors=None,
        )
        ids = tokens.get("input_ids") or []
        if len(ids) < 16:
            continue
        encoded.append(torch.tensor(ids, dtype=torch.long))
    if not encoded:
        raise SystemExit("no tokenized training samples")

    def collate(batch):
        max_len = max(item.numel() for item in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, item in enumerate(batch):
            input_ids[idx, : item.numel()] = item
            attention_mask[idx, : item.numel()] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    step = 0
    accum = max(int(args.grad_accum), 1)
    max_steps = max(int(args.max_steps), 1)
    last_loss = None
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss / accum
            loss.backward()
            last_loss = float(loss.detach().cpu()) * accum
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= max_steps:
                break
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "model_name": "BB-FinQuant-Expert-14B",
        "base_model": args.base_model,
        "inference_base_model": args.inference_base_model,
        "adapter_path": str(output_dir),
        "lora_adapter": str(output_dir),
        "specialization_id": "BB-FinQuant-Expert-14B-v1",
        "specialization_status": "trained_shadow_not_live",
        "training_artifact": str(output_dir),
        "dataset": str(dataset_path),
        "sample_count": len(rows),
        "tokenized_sample_count": len(encoded),
        "max_steps": max_steps,
        "trainable_parameters": int(trainable),
        "last_loss": last_loss,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def sh(value: str | Path) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_compact(value: Any, limit: int = 1800) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text[:limit]


def _json_object_from_remote_output(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    text = str(raw or "")
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
    if parsed is None:
        raise ValueError(f"remote output did not contain a JSON object: {text[:1000]}")
    return parsed


def _trade_response(sample: dict[str, Any]) -> dict[str, Any]:
    realized = _safe_float(sample.get("realized_pnl"))
    fees = _safe_float(sample.get("fee_estimate")) + abs(_safe_float(sample.get("funding_fee")))
    side = str(sample.get("side") or "").lower()
    return {
        "verdict": "good_trade" if realized > 0 else "bad_trade" if realized < 0 else "flat_trade",
        "side": side,
        "after_fee_pnl_usdt": round(realized, 6),
        "fee_and_funding_usdt": round(fees, 6),
        "lesson": sample.get("improvement_summary")
        or sample.get("mistake_summary")
        or (
            "Prefer similar setups only when expected net profit, liquidity, and exit discipline are stronger."
            if realized <= 0
            else "This setup had positive after-fee outcome; reuse only with comparable evidence and risk control."
        ),
        "risk_guidance": {
            "increase_size": bool(realized > 0 and _safe_float(sample.get("hold_minutes")) > 0),
            "avoid_tiny_fee_drag": bool(fees > abs(realized) and realized <= 0),
            "requires_after_fee_positive_expectancy": True,
        },
    }


def _shadow_response(sample: dict[str, Any]) -> dict[str, Any]:
    long_return = _safe_float(sample.get("long_return_pct"))
    short_return = _safe_float(sample.get("short_return_pct"))
    best_side = "long" if long_return >= short_return else "short"
    return {
        "verdict": "missed_opportunity" if sample.get("missed_opportunity") else "shadow_observation",
        "best_side": best_side,
        "long_return_pct": round(long_return, 6),
        "short_return_pct": round(short_return, 6),
        "lesson": "Rank future candidates by after-cost payoff and avoid suppressing high-quality opportunities without explicit counter-evidence.",
        "risk_guidance": {
            "do_not_trade_without_confirmation": True,
            "use_as_shadow_supervision": True,
        },
    }


def _messages(kind: str, payload: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    system = (
        "You are BB-FinQuant-Expert-14B, a cryptocurrency futures expert. "
        "Learn from audited after-fee outcomes. Reply only as compact JSON with "
        "verdict, side or best_side, lesson, and risk_guidance."
    )
    user = {
        "task": "learn_finquant_trade_policy",
        "sample_kind": kind,
        "payload": payload,
    }
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_compact(user)},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False, sort_keys=True)},
        ],
        "metadata": {
            "kind": kind,
            "symbol": payload.get("symbol"),
            "side": payload.get("side") or payload.get("decision_action"),
            "source_id": payload.get("id"),
        },
    }


async def _load_expert_memory_examples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(ExpertMemory)
            .where(ExpertMemory.is_active.is_(True))
            .order_by(ExpertMemory.confidence_score.desc(), ExpertMemory.id.desc())
            .limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())
    examples: list[dict[str, Any]] = []
    for row in rows:
        payload = {
            "id": int(row.id or 0),
            "expert_name": row.expert_name,
            "symbol": row.symbol,
            "side": row.side,
            "market_pattern": row.market_pattern,
            "confidence_score": _safe_float(row.confidence_score),
            "success_count": int(row.success_count or 0),
            "failure_count": int(row.failure_count or 0),
        }
        response = {
            "verdict": "expert_memory_lesson",
            "side": row.side,
            "lesson": row.lesson,
            "risk_guidance": {
                "recommended_action": row.recommended_action,
                "confidence_adjustment": _safe_float(row.confidence_adjustment),
                "position_size_multiplier": _safe_float(row.position_size_multiplier, 1.0),
            },
        }
        examples.append(_messages("expert_memory", payload, response))
    return examples


def _db_kind() -> str:
    url = str(settings.database_url or "").lower()
    if "postgresql" in url:
        return "postgresql"
    if "sqlite" in url:
        return "sqlite"
    return "unknown"


async def _missing_training_tables() -> list[str]:
    expected = set(REQUIRED_TRAINING_TABLES)
    async with get_session_ctx() as session:
        if _db_kind() == "sqlite":
            result = await session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            present = {str(row[0]) for row in result.fetchall()} & expected
        elif _db_kind() == "postgresql":
            result = await session.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    """
                )
            )
            present = {str(row[0]) for row in result.fetchall()} & expected
        else:
            present = set()
    return sorted(expected - present)


async def _training_data_source_ready() -> tuple[bool, list[str]]:
    try:
        missing = await _missing_training_tables()
    except SQLAlchemyError:
        raise
    return (not missing, missing)


async def _assert_training_data_source_ready() -> None:
    ready, missing = await _training_data_source_ready()
    if ready:
        return
    raise RuntimeError(
        "BB-FinQuant-Expert-14B training data source is not ready: "
        f"missing tables {missing}. Use --source platform or run this script on "
        "the online platform server where the trading PostgreSQL database is configured."
    )


async def build_dataset(
    *,
    trade_limit: int,
    shadow_limit: int,
    memory_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    await _assert_training_data_source_ready()
    reflections = await _load_trade_reflection_samples(trade_limit)
    authoritative = await _load_authoritative_trade_samples(trade_limit)
    trade_samples = _merge_trade_samples(reflections, authoritative)
    shadow_samples = await _load_shadow_samples(shadow_limit)
    quality = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    examples: list[dict[str, Any]] = []
    for sample in quality["trade_samples"]:
        if str(sample.get("trade_fact_trust_reason") or "").strip():
            continue
        payload = {
            key: sample.get(key)
            for key in (
                "id",
                "position_id",
                "symbol",
                "side",
                "entry_price",
                "exit_price",
                "quantity",
                "realized_pnl",
                "fee_estimate",
                "funding_fee",
                "hold_minutes",
                "leverage",
                "outcome",
                "raw_llm_response",
                "settlement_status",
                "settlement_source",
            )
            if key in sample
        }
        examples.append(_messages("closed_trade", payload, _trade_response(sample)))
    for sample in quality["shadow_samples"]:
        payload = {
            key: sample.get(key)
            for key in (
                "id",
                "symbol",
                "analysis_type",
                "decision_action",
                "decision_confidence",
                "horizon_minutes",
                "features",
                "long_return_pct",
                "short_return_pct",
                "best_action",
                "missed_opportunity",
            )
            if key in sample
        }
        examples.append(_messages("shadow_backtest", payload, _shadow_response(sample)))
    examples.extend(await _load_expert_memory_examples(memory_limit))
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_policy": "bb_finquant_expert_sft_v1",
        "trade_limit": trade_limit,
        "shadow_limit": shadow_limit,
        "memory_limit": memory_limit,
        "example_count": len(examples),
        "quality_report": quality.get("quality_report", {}),
        "source": "platform_db_clean_training_view",
    }
    return examples, manifest


def _load_old_server_info(account_dir: Path):
    path = account_dir / OLD_PROFILE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"old model-server file not found: {path}")
    return parse_remote_server_info(path.read_text(encoding="utf-8", errors="replace"), source_path=path)


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    run_remote_text(ssh, f"mkdir -p {sh(posixpath.dirname(remote_path))}", timeout=30)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _download_text(ssh, remote_path: str) -> str:
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "r") as remote:
            data = remote.read()
    finally:
        sftp.close()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def export_dataset_from_platform(
    *,
    trade_limit: int,
    shadow_limit: int,
    memory_limit: int,
) -> tuple[str, str, dict[str, Any]]:
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        _upload_text(
            ssh,
            REMOTE_PLATFORM_SCRIPT,
            Path(__file__).read_text(encoding="utf-8"),
            mode=0o755,
        )
        wrapper = f"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path({REMOTE_PLATFORM_APP_DIR!r})

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip(chr(34)).strip(chr(39))
        if key:
            os.environ[key] = value

os.chdir(ROOT)
load_env(ROOT / ".env")
load_env(Path("/etc/bb/bb-runtime.env"))
sys.argv = [
    "scripts/finquant_expert_lora_training.py",
    "--source",
    "local",
    "--export-only",
    "--trade-limit",
    {str(int(trade_limit))!r},
    "--shadow-limit",
    {str(int(shadow_limit))!r},
    "--memory-limit",
    {str(int(memory_limit))!r},
]
runpy.run_path("scripts/finquant_expert_lora_training.py", run_name="__main__")
"""
        _upload_text(ssh, REMOTE_PLATFORM_EXPORT_WRAPPER, wrapper, mode=0o755)
        run_remote_text(
            ssh,
            f"chown -R bb:bb {sh(REMOTE_PLATFORM_EXPORT_DIR)} 2>/dev/null || true",
            timeout=30,
            check=False,
        )
        inner_command = (
            f"cd {sh(REMOTE_PLATFORM_APP_DIR)} && "
            "PYBIN=python3; "
            "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
            "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
            f"$PYBIN {sh(REMOTE_PLATFORM_EXPORT_WRAPPER)}"
        )
        command = (
            "if id -u bb >/dev/null 2>&1; then "
            f"sudo -u bb -H bash -lc {sh(inner_command)}; "
            "else "
            f"{inner_command}; "
            "fi"
        )
        raw = run_remote_text(ssh, command, timeout=600, check=True)
        dataset_jsonl = _download_text(
            ssh,
            f"{REMOTE_PLATFORM_EXPORT_DIR}/finquant_sft_latest.jsonl",
        )
        manifest_json = _download_text(
            ssh,
            f"{REMOTE_PLATFORM_EXPORT_DIR}/finquant_sft_latest_manifest.json",
        )
        remote_result: dict[str, Any] = {}
        try:
            remote_result = json.loads(raw)
        except json.JSONDecodeError:
            remote_result = {"raw": raw[:1200]}
        return dataset_jsonl, manifest_json, remote_result
    finally:
        ssh.close()


def _remote_service_update_script(*, adapter_path: str | None) -> str:
    lora_args = ""
    upstream_model = BASE_MODEL_NAME
    if adapter_path:
        lora_args = (
            "  --enable-lora \\\n"
            f"  --lora-modules {MODEL_NAME}={adapter_path} \\\n"
        )
        upstream_model = MODEL_NAME
    qwen_script = f"""#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate trade_vllm
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/trade_ai/hf_cache
export HF_HUB_CACHE=/data/trade_ai/hf_cache/hub
export TRANSFORMERS_CACHE=/data/trade_ai/hf_cache/transformers
LOG=/data/trade_ai/logs/finquant_expert_14b.log
exec python -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --model {REMOTE_INFERENCE_BASE_MODEL} \\
  --served-model-name {BASE_MODEL_NAME} \\
  --trust-remote-code \\
  --max-model-len 4096 \\
  --gpu-memory-utilization 0.72 \\
  --dtype half \\
  --quantization awq_marlin \\
  --max-num-seqs 2 \\
  --max-num-batched-tokens 4096 \\
{lora_args}  --enable-prefix-caching \\
  --enable-chunked-prefill > "$LOG" 2>&1
"""
    alias_script = f'''from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 8003
UPSTREAM = "http://127.0.0.1:8000"
ALIAS_MODEL = "{MODEL_NAME}"
UPSTREAM_MODEL = "{upstream_model}"
FALLBACK_MODEL = "{BASE_MODEL_NAME}"


class Handler(BaseHTTPRequestHandler):
    server_version = "BBFinQuantAlias/2.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\\n" % (self.address_string(), fmt % args))

    def _write_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._write_json(200, {{"object": "list", "data": [{{
                "id": ALIAS_MODEL,
                "object": "model",
                "owned_by": "bb-finquant-specialization",
                "root": UPSTREAM_MODEL,
                "parent": FALLBACK_MODEL,
            }}]}})
            return
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def _proxy(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        headers = {{key: value for key, value in self.headers.items() if key.lower() not in {{"host", "content-length", "connection"}}}}
        if self.path.startswith("/v1/") and body:
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("model") == ALIAS_MODEL:
                    payload["model"] = UPSTREAM_MODEL
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    headers["content-type"] = "application/json"
            except Exception:
                pass
        request = urllib.request.Request(UPSTREAM + self.path, data=body if self.command != "GET" else None, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in {{"transfer-encoding", "connection"}}:
                        continue
                    self.send_header(key, value)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("content-type", exc.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._write_json(502, {{"error": "finquant_alias_upstream_failed", "detail": str(exc)[:200]}})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
'''
    return textwrap.dedent(
        f"""
        set -euo pipefail
        cp {sh(REMOTE_QWEN_START_SCRIPT)} {sh(REMOTE_QWEN_START_SCRIPT + '.bak.' + str(int(time.time())))}
        cat > {sh(REMOTE_QWEN_START_SCRIPT)} <<'SH'
{qwen_script.rstrip()}
SH
        chmod +x {sh(REMOTE_QWEN_START_SCRIPT)}
        mkdir -p {sh(posixpath.dirname(REMOTE_ALIAS_SCRIPT))}
        cat > {sh(REMOTE_ALIAS_SCRIPT)} <<'PY'
{alias_script.rstrip()}
PY
        chmod +x {sh(REMOTE_ALIAS_SCRIPT)}
        sudo -n systemctl restart qwen3-14b-trade.service
        sudo -n systemctl restart bb-finquant-expert-alias.service
        """
    ).strip()


def deploy_and_optionally_train(
    *,
    account_dir: Path,
    dataset_jsonl: str,
    manifest_json: str,
    train: bool,
    switch_service: bool,
    stop_inference_for_training: bool,
    max_steps: int,
    max_samples: int,
) -> dict[str, Any]:
    info = _load_old_server_info(account_dir)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, REMOTE_DATASET, dataset_jsonl)
        _upload_text(ssh, REMOTE_DATASET_MANIFEST, manifest_json)
        _upload_text(ssh, REMOTE_TRAINER, REMOTE_TRAINER_CODE, mode=0o755)
        result: dict[str, Any] = {
            "uploaded": {
                "dataset": REMOTE_DATASET,
                "dataset_manifest": REMOTE_DATASET_MANIFEST,
                "trainer": REMOTE_TRAINER,
            },
            "trained": False,
            "service_switched": False,
        }
        if train:
            pre = ""
            post = ""
            if stop_inference_for_training:
                pre = "sudo -n systemctl stop bb-finquant-expert-alias.service qwen3-14b-trade.service deepseek-r1-14b-risk.service || true; "
                post = "sudo -n systemctl start qwen3-14b-trade.service bb-finquant-expert-alias.service deepseek-r1-14b-risk.service || true; "
            train_cmd = (
                "set -euo pipefail; "
                f"mkdir -p {sh(REMOTE_TRAINING_DIR)} {sh(REMOTE_ADAPTER_DIR)}; "
                f"{_remote_train_base_prepare_command()}; "
                f"{pre}"
                f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_TRAINER)} "
                f"--dataset {sh(REMOTE_DATASET)} "
                f"--output-dir {sh(REMOTE_ADAPTER_DIR)} "
                f"--base-model {sh(REMOTE_TRAIN_BASE_MODEL)} "
                f"--inference-base-model {sh(REMOTE_INFERENCE_BASE_MODEL)} "
                f"--manifest {sh(REMOTE_SPECIALIZATION_MANIFEST)} "
                f"--max-steps {int(max_steps)} --max-samples {int(max_samples)} "
                f"> {sh(REMOTE_TRAIN_LOG)} 2>&1; "
                f"{post}"
                f"cat {sh(REMOTE_SPECIALIZATION_MANIFEST)}"
            )
            try:
                raw = run_remote_text(ssh, train_cmd, timeout=7200, check=True)
            except Exception as exc:
                tail = run_remote_text(
                    ssh,
                    f"tail -n 160 {sh(REMOTE_TRAIN_LOG)} 2>/dev/null || true; "
                    "sudo -n systemctl start qwen3-14b-trade.service "
                    "bb-finquant-expert-alias.service deepseek-r1-14b-risk.service || true",
                    timeout=120,
                    check=False,
                )
                raise RuntimeError(
                    "remote LoRA training failed. "
                    f"Command error: {safe_error_text(exc, limit=1400)}\n"
                    f"Train log tail:\n{tail}"
                ) from None
            result["trained"] = True
            result["specialization_manifest"] = _json_object_from_remote_output(raw)
            run_remote_text(ssh, _remote_manifest_update_command(), timeout=120, check=True)
        if switch_service:
            adapter = REMOTE_ADAPTER_DIR if train or _remote_manifest_exists(ssh) else None
            run_remote_text(
                ssh,
                _remote_service_update_script(adapter_path=adapter),
                timeout=180,
                check=True,
            )
            probe = run_remote_text(
                ssh,
                "set +e; "
                "alias_payload=$(curl -fsS --max-time 5 http://127.0.0.1:8003/v1/models 2>&1); "
                "upstream_payload=''; "
                "for i in $(seq 1 120); do "
                "  upstream_payload=$(curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models 2>&1); "
                "  rc=$?; "
                "  if [ \"$rc\" -eq 0 ]; then break; fi; "
                "  sleep 3; "
                "done; "
                "printf '%s\\n---8000---\\n%s\\n' \"$alias_payload\" \"$upstream_payload\"",
                timeout=420,
                check=False,
            )
            result["service_switched"] = True
            result["probe"] = probe[:2000]
            alias_probe, _separator, upstream_probe = probe.partition("---8000---")
            if MODEL_NAME not in alias_probe or BASE_MODEL_NAME not in upstream_probe:
                raise RuntimeError(
                    "BB-FinQuant service switch did not verify both alias and upstream "
                    f"models. Probe excerpt:\n{probe[:2000]}"
                )
            if adapter and MODEL_NAME not in upstream_probe:
                raise RuntimeError(
                    "BB-FinQuant adapter was requested but upstream vLLM did not expose "
                    f"the LoRA model. Probe excerpt:\n{probe[:2000]}"
                )
        return result
    finally:
        ssh.close()


def _remote_manifest_exists(ssh) -> bool:
    raw = run_remote_text(
        ssh,
        f"test -s {sh(REMOTE_SPECIALIZATION_MANIFEST)} && echo yes || echo no",
        timeout=20,
        check=False,
    )
    return raw.strip() == "yes"


def _remote_train_base_prepare_command() -> str:
    return textwrap.dedent(
        f"""
        if [ -s {sh(REMOTE_TRAIN_BASE_MODEL + "/config.json")} ]; then
          echo train-base-ready
        else
          mkdir -p {sh(REMOTE_TRAIN_BASE_MODEL)}
          /data/BB/envs/phase3-quant/bin/python - <<'PY'
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id={REMOTE_TRAIN_BASE_REPO!r},
    local_dir={REMOTE_TRAIN_BASE_MODEL!r},
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=4,
    ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot"],
)
print("train-base-downloaded")
PY
        fi
        """
    ).strip()


def _remote_manifest_update_command() -> str:
    return textwrap.dedent(
        f"""
        /data/BB/envs/phase3-quant/bin/python - <<'PY'
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

adapter_manifest_path = Path({REMOTE_SPECIALIZATION_MANIFEST!r})
paths = [
    Path({REMOTE_DOWNLOAD_MANIFEST!r}),
    Path({REMOTE_VALIDATION_MANIFEST!r}),
]
if not adapter_manifest_path.exists():
    raise SystemExit(f"missing specialization manifest: {{adapter_manifest_path}}")
adapter = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
evidence = {{
    "served_model_name": {MODEL_NAME!r},
    "specialization_required": True,
    "specialization_target": {MODEL_NAME!r},
    "specialization_status": adapter.get("specialization_status") or "trained_shadow_not_live",
    "base_model_carrier": {REMOTE_INFERENCE_BASE_MODEL!r},
    "adapter_path": adapter.get("adapter_path") or {REMOTE_ADAPTER_DIR!r},
    "lora_adapter": adapter.get("lora_adapter") or {REMOTE_ADAPTER_DIR!r},
    "specialization_manifest": str(adapter_manifest_path),
    "specialization_id": adapter.get("specialization_id") or "BB-FinQuant-Expert-14B-v1",
    "training_artifact": adapter.get("training_artifact") or {REMOTE_ADAPTER_DIR!r},
    "specialization_evidence": {{
        "adapter_path": adapter.get("adapter_path") or {REMOTE_ADAPTER_DIR!r},
        "lora_adapter": adapter.get("lora_adapter") or {REMOTE_ADAPTER_DIR!r},
        "specialization_manifest": str(adapter_manifest_path),
        "training_artifact": adapter.get("training_artifact") or {REMOTE_ADAPTER_DIR!r},
        "trained_at": adapter.get("trained_at"),
        "sample_count": adapter.get("sample_count"),
        "max_steps": adapter.get("max_steps"),
    }},
}}
for path in paths:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {{"models": []}}
    models = data.setdefault("models", [])
    target = None
    for row in models:
        if not isinstance(row, dict):
            continue
        if row.get("slot") == "llm_expert_pool" or row.get("served_model_name") == {MODEL_NAME!r}:
            target = row
            break
    if target is None:
        target = {{
            "slot": "llm_expert_pool",
            "repo_id": "Qwen/Qwen3-14B-AWQ",
            "path": {REMOTE_INFERENCE_BASE_MODEL!r},
            "target": {REMOTE_INFERENCE_BASE_MODEL!r},
            "role": "expert_pool",
            "status": "ready",
            "exists": True,
        }}
        models.append(target)
    target.update(evidence)
    data["checked_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({{"updated": [str(path) for path in paths], "evidence": evidence}}, ensure_ascii=False))
PY
        """
    ).strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-dir", type=Path, default=DEFAULT_ACCOUNT_INFO_DIR)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "platform"),
        default="auto",
        help=(
            "Training data source. auto uses the local DB when it has the required "
            "training tables, otherwise exports from the online platform server."
        ),
    )
    parser.add_argument("--trade-limit", type=int, default=900)
    parser.add_argument("--shadow-limit", type=int, default=1200)
    parser.add_argument("--memory-limit", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--switch-service", action="store_true")
    parser.add_argument("--stop-inference-for-training", action="store_true")
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        try:
            local_ready, _missing = await _training_data_source_ready()
        except SQLAlchemyError:
            local_ready = False
        source = "local" if local_ready else "platform"

    if source == "platform":
        dataset_jsonl, manifest_json, platform_export = export_dataset_from_platform(
            trade_limit=args.trade_limit,
            shadow_limit=args.shadow_limit,
            memory_limit=args.memory_limit,
        )
        try:
            manifest = json.loads(manifest_json)
        except json.JSONDecodeError:
            manifest = {}
        example_count = int(manifest.get("example_count") or 0)
        if example_count <= 0:
            example_count = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
        manifest["source_transport"] = "online_platform_export"
        manifest["platform_export"] = platform_export
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    else:
        examples, manifest = await build_dataset(
            trade_limit=args.trade_limit,
            shadow_limit=args.shadow_limit,
            memory_limit=args.memory_limit,
        )
        if not examples:
            raise SystemExit("No BB-FinQuant SFT examples were generated.")
        dataset_jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
        manifest["source_transport"] = "local_database"
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        example_count = len(examples)

    if not dataset_jsonl.strip():
        raise SystemExit("No BB-FinQuant SFT examples were generated.")
    local_dir = ROOT / "data" / "finquant_expert_training"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "finquant_sft_latest.jsonl").write_text(dataset_jsonl, encoding="utf-8")
    (local_dir / "finquant_sft_latest_manifest.json").write_text(manifest_json, encoding="utf-8")
    if args.export_only:
        safe_print(
            json.dumps(
                {
                    "exported": True,
                    "source": source,
                    "local_dataset": str(local_dir / "finquant_sft_latest.jsonl"),
                    "local_manifest": str(local_dir / "finquant_sft_latest_manifest.json"),
                    "example_count": example_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = deploy_and_optionally_train(
        account_dir=args.account_dir,
        dataset_jsonl=dataset_jsonl,
        manifest_json=manifest_json,
        train=args.train,
        switch_service=args.switch_service,
        stop_inference_for_training=args.stop_inference_for_training,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
    )
    result["source"] = source
    result["example_count"] = example_count
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
