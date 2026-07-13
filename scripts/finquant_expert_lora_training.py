#!/usr/bin/env python3
"""Export BB-FinQuant SFT data and run model-server LoRA specialization.

This script deliberately treats BB-FinQuant-Expert-14B specialization as a real
artifact-producing training job. A renamed Qwen endpoint is not considered
trained unless an adapter directory and specialization manifest are produced.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import posixpath
import re
import subprocess
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
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
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

REMOTE_ROOT = "/data/BB"
REMOTE_TRAINING_DIR = f"{REMOTE_ROOT}/training/finquant_expert"
REMOTE_SERVICE_DIR = f"{REMOTE_ROOT}/services/finquant_expert_training"
REMOTE_DATASET_VERSIONS_DIR = f"{REMOTE_TRAINING_DIR}/versions"
REMOTE_DATASET_CURRENT = f"{REMOTE_TRAINING_DIR}/current.json"
REMOTE_TRAINER = f"{REMOTE_SERVICE_DIR}/train_finquant_lora.py"
REMOTE_REGISTRY_TOOL = f"{REMOTE_SERVICE_DIR}/finquant_registry.py"
REMOTE_ADAPTER_ROOT = f"{REMOTE_ROOT}/models/finquant_lora"
REMOTE_ADAPTER_VERSIONS_DIR = f"{REMOTE_ADAPTER_ROOT}/versions"
REMOTE_ADAPTER_CURRENT = f"{REMOTE_ADAPTER_ROOT}/current.json"
REMOTE_ADAPTER_ROLLBACK = f"{REMOTE_ADAPTER_ROOT}/rollback.json"
REMOTE_LEGACY_ADAPTER_DIR = f"{REMOTE_ADAPTER_ROOT}/BB-FinQuant-Expert-14B-v1"
REMOTE_LEGACY_SPECIALIZATION_MANIFEST = f"{REMOTE_LEGACY_ADAPTER_DIR}/specialization_manifest.json"
REMOTE_INFERENCE_BASE_MODEL = "/data/trade_models/Qwen/Qwen3-14B-AWQ"
REMOTE_TRAIN_BASE_REPO = "Qwen/Qwen3-14B"
REMOTE_TRAIN_BASE_MODEL = f"{REMOTE_ROOT}/models/trainable/Qwen3-14B"
REMOTE_QWEN_START_SCRIPT = "/data/trade_ai/scripts/start_qwen3_14b_trade.sh"
REMOTE_GATEWAY_DIR = f"{REMOTE_ROOT}/services/finquant_expert_gateway"
REMOTE_GATEWAY_SCRIPT = f"{REMOTE_GATEWAY_DIR}/gateway.py"
REMOTE_GATEWAY_SERVICE = "bb-finquant-expert-gateway.service"
REMOTE_GATEWAY_SERVICE_PATH = f"/etc/systemd/system/{REMOTE_GATEWAY_SERVICE}"
REMOTE_LEGACY_ALIAS_SERVICE = "bb-finquant-expert-alias.service"
REMOTE_TRAIN_LOG_DIR = f"{REMOTE_TRAINING_DIR}/logs"
REMOTE_DOWNLOAD_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_download_manifest.json"
REMOTE_VALIDATION_MANIFEST = f"{REMOTE_ROOT}/manifests/phase3_model_validation.json"
REMOTE_PLATFORM_APP_DIR = "/data/bb/app"
REMOTE_PLATFORM_SCRIPT = f"{REMOTE_PLATFORM_APP_DIR}/scripts/finquant_expert_lora_training.py"
REMOTE_PLATFORM_EXPORT_DIR = f"{REMOTE_PLATFORM_APP_DIR}/data/finquant_expert_training"
REMOTE_PLATFORM_EXPORT_WRAPPER = f"{REMOTE_PLATFORM_EXPORT_DIR}/export_wrapper.py"
MODEL_NAME = "BB-FinQuant-Expert-14B"
BASE_MODEL_NAME = "qwen3-14b-trade"
DATASET_SCHEMA_VERSION = "bb_finquant_expert_sft.v2"
RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"
RETURN_OBJECTIVE_VERSION = "2026-07-12.v1"
PREFERENCE_CONTRACT_VERSION = "bb_finquant_return_preference.v1"
ADAPTER_REGISTRY_VERSION = "bb_finquant_lora.v2"
DATASET_VERSION_PATTERN = re.compile(r"^bb-finquant-sft-v2-[0-9a-f]{12}-[0-9a-f]{8}$")
ADAPTER_VERSION_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
REQUIRED_TRAINING_TABLES = (
    "trade_reflections",
    "positions",
    "orders",
    "shadow_backtests",
    "expert_memories",
)


REMOTE_TRAINER_CODE = r"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_hash(row: dict) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 3:
                    raise SystemExit("invalid SFT message contract")
                for message in messages:
                    if not isinstance(message, dict):
                        raise SystemExit("invalid SFT message row")
                    if message.get("role") in {"user", "assistant"}:
                        try:
                            parsed_content = json.loads(str(message.get("content") or ""))
                        except json.JSONDecodeError as exc:
                            raise SystemExit(
                                "SFT user/assistant content is not valid JSON"
                            ) from exc
                        if not isinstance(parsed_content, dict):
                            raise SystemExit(
                                "SFT user/assistant JSON content must be an object"
                            )
                rows.append(row)
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


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(rows) < 10:
        return rows, []
    ranked = sorted((canonical_row_hash(row), index) for index, row in enumerate(rows))
    eval_count = min(max(len(rows) // 10, 1), 64)
    eval_indices = {item[1] for item in ranked[:eval_count]}
    train_rows = [row for index, row in enumerate(rows) if index not in eval_indices]
    eval_rows = [row for index, row in enumerate(rows) if index in eval_indices]
    return train_rows, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-repo", required=True)
    parser.add_argument("--inference-base-model", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_manifest_path = Path(args.dataset_manifest)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite adapter version: {output_dir}")
    if manifest_path.parent != output_dir:
        raise SystemExit("specialization manifest must be stored inside its adapter version")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_manifest.get("dataset_sha256") != dataset_sha256:
        raise SystemExit("dataset SHA-256 does not match its manifest")
    if dataset_manifest.get("dataset_schema_version") != "bb_finquant_expert_sft.v2":
        raise SystemExit("unsupported BB-FinQuant dataset schema")
    if dataset_manifest.get("objective_name") != "maximize_expected_realized_net_return_after_cost":
        raise SystemExit("dataset return objective mismatch")
    if dataset_manifest.get("objective_version") != "2026-07-12.v1":
        raise SystemExit("dataset return objective version mismatch")
    base_identity = dataset_manifest.get("base_model_identity")
    if not isinstance(base_identity, dict) or base_identity.get("training_repo") != args.base_model_repo:
        raise SystemExit("dataset and trainer base-model identities do not match")
    for key in ("dataset_lineage_sha256", "source_script_sha256"):
        value = str(dataset_manifest.get(key) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise SystemExit(f"dataset manifest has no valid {key}")
    rows = load_rows(dataset_path)
    if not rows:
        raise SystemExit("empty training dataset")
    train_rows, eval_rows = split_rows(rows)
    preference_rows = []
    for row in train_rows:
        preference = row.get("preference")
        if not isinstance(preference, dict):
            continue
        if preference.get("contract_version") != "bb_finquant_return_preference.v1":
            raise SystemExit("unsupported return preference contract")
        if preference.get("objective") != "maximize_expected_realized_net_return_after_cost":
            raise SystemExit("return preference objective mismatch")
        preference_rows.append({
            "prompt": str(preference.get("prompt") or ""),
            "chosen": str(preference.get("chosen") or ""),
            "rejected": str(preference.get("rejected") or ""),
        })
    if not preference_rows:
        raise SystemExit("no return-preference rows available for DPO")

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

    def encode_rows(source_rows):
        encoded_rows = []
        for row in source_rows:
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
            encoded_rows.append(torch.tensor(ids, dtype=torch.long))
        return encoded_rows

    encoded = encode_rows(train_rows)
    eval_encoded = encode_rows(eval_rows)
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
    pending_accumulation = 0
    optimizer_steps = 0
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss / accum
            loss.backward()
            last_loss = float(loss.detach().cpu()) * accum
            pending_accumulation += 1
            if pending_accumulation >= accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_accumulation = 0
            step += 1
            if step >= max_steps:
                break
    if pending_accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    eval_loss = None
    if eval_encoded:
        model.eval()
        eval_loader = DataLoader(
            eval_encoded,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                eval_losses.append(float(model(**batch).loss.detach().cpu()))
        if eval_losses:
            eval_loss = sum(eval_losses) / len(eval_losses)

    try:
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:
        raise SystemExit(f"TRL DPO runtime unavailable: {type(exc).__name__}: {exc}") from exc

    dpo_output_dir = output_dir.parent / f".{output_dir.name}.dpo-work"
    dpo_config = DPOConfig(
        output_dir=str(dpo_output_dir),
        per_device_train_batch_size=max(int(args.batch_size), 1),
        gradient_accumulation_steps=max(int(args.grad_accum), 1),
        max_steps=max(min(max_steps // 2, 40), 10),
        learning_rate=float(args.learning_rate) * 0.5,
        beta=0.10,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    dpo_kwargs = {
        "model": model,
        "args": dpo_config,
        "train_dataset": Dataset.from_list(preference_rows),
        "processing_class": tokenizer,
    }
    try:
        dpo_trainer = DPOTrainer(**dpo_kwargs)
    except TypeError:
        dpo_kwargs["tokenizer"] = dpo_kwargs.pop("processing_class")
        dpo_trainer = DPOTrainer(**dpo_kwargs)
    dpo_result = dpo_trainer.train()
    dpo_train_loss = float(getattr(dpo_result, "training_loss", 0.0) or 0.0)
    preference_selection_accuracy = None
    for log_row in reversed(list(getattr(dpo_trainer.state, "log_history", []) or [])):
        for key in ("rewards/accuracies", "eval_rewards/accuracies"):
            if key in log_row:
                preference_selection_accuracy = float(log_row[key])
                break
        if preference_selection_accuracy is not None:
            break
    shutil.rmtree(dpo_output_dir, ignore_errors=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging.{os.getpid()}")
    if staging_dir.exists():
        raise SystemExit(f"adapter staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(staging_dir)
    tokenizer.save_pretrained(staging_dir)
    evaluation_path = staging_dir / "evaluation_report.json"
    evaluation = {
        "status": "shadow_only_not_promotion_evidence",
        "training_micro_steps": step,
        "optimizer_steps": optimizer_steps,
        "train_sample_count": len(train_rows),
        "tokenized_train_sample_count": len(encoded),
        "eval_sample_count": len(eval_rows),
        "tokenized_eval_sample_count": len(eval_encoded),
        "last_train_loss": last_loss,
        "held_out_eval_loss": eval_loss,
        "held_out_eval_loss_role": "format_and_language_fit_only_not_profit_evidence",
        "preference_contract_version": "bb_finquant_return_preference.v1",
        "preference_train_count": len(preference_rows),
        "dpo_train_loss": dpo_train_loss,
        "preference_selection_accuracy": preference_selection_accuracy,
        "return_objective": "maximize_expected_realized_net_return_after_cost",
        "training_stages": ["sft_format_domain", "trl_dpo_return_preference"],
    }
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    adapter_files = []
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file():
            continue
        adapter_files.append({
            "path": path.relative_to(staging_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    adapter_digest_payload = json.dumps(
        adapter_files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    adapter_sha256 = hashlib.sha256(adapter_digest_payload).hexdigest()
    base_config_path = Path(args.base_model) / "config.json"
    inference_config_path = Path(args.inference_base_model) / "config.json"
    manifest = {
        "registry_version": "bb_finquant_lora.v2",
        "model_name": "BB-FinQuant-Expert-14B",
        "adapter_version": args.version_id,
        "base_model": args.base_model,
        "base_model_repo": args.base_model_repo,
        "base_model_config_sha256": (
            sha256_file(base_config_path) if base_config_path.exists() else None
        ),
        "inference_base_model": args.inference_base_model,
        "inference_base_model_config_sha256": (
            sha256_file(inference_config_path) if inference_config_path.exists() else None
        ),
        "adapter_path": str(output_dir),
        "lora_adapter": str(output_dir),
        "specialization_id": f"BB-FinQuant-Expert-14B-{args.version_id}",
        "specialization_status": "trained_shadow_not_live",
        "training_artifact": str(output_dir),
        "dataset": str(dataset_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_schema_version": dataset_manifest.get("dataset_schema_version"),
        "dataset_version": dataset_manifest.get("dataset_version"),
        "dataset_sha256": dataset_sha256,
        "dataset_lineage_sha256": dataset_manifest.get("dataset_lineage_sha256"),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source_code_version": dataset_manifest.get("source_code_version"),
        "source_script_sha256": dataset_manifest.get("source_script_sha256"),
        "trainer_code_sha256": sha256_file(Path(__file__)),
        "sample_count": len(rows),
        "train_sample_count": len(train_rows),
        "eval_sample_count": len(eval_rows),
        "tokenized_sample_count": len(encoded),
        "max_steps": max_steps,
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": int(trainable),
        "last_loss": last_loss,
        "held_out_eval_loss": eval_loss,
        "held_out_eval_loss_role": "format_and_language_fit_only_not_profit_evidence",
        "preference_contract_version": "bb_finquant_return_preference.v1",
        "preference_train_count": len(preference_rows),
        "dpo_train_loss": dpo_train_loss,
        "preference_selection_accuracy": preference_selection_accuracy,
        "objective_name": "maximize_expected_realized_net_return_after_cost",
        "objective_version": "2026-07-12.v1",
        "training_stages": ["sft_format_domain", "trl_dpo_return_preference"],
        "evaluation_report": str(output_dir / evaluation_path.name),
        "training_config": {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "batch_size": args.batch_size,
            "gradient_accumulation": accum,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "torch_dtype": "float16",
            "gradient_checkpointing": True,
        },
        "adapter_files": adapter_files,
        "adapter_sha256": adapter_sha256,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    staging_manifest_path = staging_dir / manifest_path.name
    staging_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(staging_dir, output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""


REMOTE_REGISTRY_TOOL_CODE = r"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_NAME = "BB-FinQuant-Expert-14B"
REGISTRY_VERSION = "bb_finquant_lora.v2"
ROOT = Path("/data/BB/models/finquant_lora")
VERSIONS = ROOT / "versions"
CURRENT = ROOT / "current.json"
ROLLBACK = ROOT / "rollback.json"
RETIRED = ROOT / "retired"
LEGACY = ROOT / "BB-FinQuant-Expert-14B-v1"
LEGACY_MANIFEST = LEGACY / "specialization_manifest.json"
DOWNLOAD_MANIFEST = Path("/data/BB/manifests/phase3_model_download_manifest.json")
VALIDATION_MANIFEST = Path("/data/BB/manifests/phase3_model_validation.json")
INFERENCE_BASE = "/data/trade_models/Qwen/Qwen3-14B-AWQ"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def adapter_digest(files: list[dict]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_pointer(pointer: dict) -> tuple[dict, Path]:
    if pointer.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("unsupported FinQuant pointer registry version")
    if pointer.get("model_name") != MODEL_NAME:
        raise ValueError("FinQuant pointer model identity mismatch")
    adapter_path = inside(Path(str(pointer.get("adapter_path") or "")), ROOT)
    manifest_path = inside(Path(str(pointer.get("manifest_path") or "")), ROOT)
    if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise ValueError("FinQuant specialization manifest hash mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError("FinQuant specialization manifest model mismatch")
    if pointer.get("legacy_read_only"):
        primary = inside(Path(str(pointer.get("primary_adapter_path") or "")), adapter_path)
        if sha256_file(primary) != pointer.get("adapter_sha256"):
            raise ValueError("legacy FinQuant adapter hash mismatch")
        return manifest, adapter_path
    if manifest.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("FinQuant adapter manifest registry mismatch")
    if manifest.get("adapter_version") != pointer.get("adapter_version"):
        raise ValueError("FinQuant adapter version mismatch")
    if manifest.get("dataset_schema_version") != "bb_finquant_expert_sft.v2":
        raise ValueError("FinQuant adapter dataset schema mismatch")
    if manifest.get("objective_name") != "maximize_expected_realized_net_return_after_cost":
        raise ValueError("FinQuant adapter return objective mismatch")
    if manifest.get("objective_version") != "2026-07-12.v1":
        raise ValueError("FinQuant adapter return objective version mismatch")
    if manifest.get("preference_contract_version") != "bb_finquant_return_preference.v1":
        raise ValueError("FinQuant adapter preference contract mismatch")
    if manifest.get("base_model_repo") != "Qwen/Qwen3-14B":
        raise ValueError("FinQuant adapter base-model identity mismatch")
    for key in (
        "dataset_sha256",
        "dataset_lineage_sha256",
        "dataset_manifest_sha256",
        "source_script_sha256",
        "trainer_code_sha256",
        "base_model_config_sha256",
        "inference_base_model_config_sha256",
    ):
        value = str(manifest.get(key) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"FinQuant adapter manifest has no valid {key}")
    if not isinstance(manifest.get("training_config"), dict):
        raise ValueError("FinQuant adapter training configuration is missing")
    files = manifest.get("adapter_files")
    if not isinstance(files, list) or not files:
        raise ValueError("FinQuant adapter file manifest is empty")
    verified_files = []
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("invalid FinQuant adapter file row")
        relative = str(row.get("path") or "")
        path = inside(adapter_path / relative, adapter_path)
        verified = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if verified != row:
            raise ValueError(f"FinQuant adapter file verification failed: {relative}")
        verified_files.append(verified)
    if not any(row["path"] in {"adapter_model.safetensors", "adapter_model.bin"} for row in verified_files):
        raise ValueError("FinQuant adapter weights are missing")
    digest = adapter_digest(verified_files)
    if digest != manifest.get("adapter_sha256") or digest != pointer.get("adapter_sha256"):
        raise ValueError("FinQuant aggregate adapter hash mismatch")
    return manifest, adapter_path


def pointer_for_manifest(manifest_path: Path) -> dict:
    manifest_path = inside(manifest_path, VERSIONS)
    manifest = read_json(manifest_path)
    adapter_path = manifest_path.parent
    pointer = {
        "registry_version": REGISTRY_VERSION,
        "model_name": MODEL_NAME,
        "adapter_version": manifest.get("adapter_version"),
        "specialization_id": manifest.get("specialization_id"),
        "adapter_path": str(adapter_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": manifest.get("adapter_sha256"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "legacy_read_only": False,
    }
    validate_pointer(pointer)
    return pointer


def legacy_pointer() -> dict | None:
    if not LEGACY_MANIFEST.exists():
        return None
    manifest = read_json(LEGACY_MANIFEST)
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError("legacy FinQuant manifest model identity mismatch")
    candidates = [LEGACY / "adapter_model.safetensors", LEGACY / "adapter_model.bin"]
    primary = next((path for path in candidates if path.is_file()), None)
    if primary is None:
        raise ValueError("legacy FinQuant adapter weights are missing")
    pointer = {
        "registry_version": REGISTRY_VERSION,
        "model_name": MODEL_NAME,
        "adapter_version": "legacy-v1-read-only",
        "specialization_id": manifest.get("specialization_id") or "BB-FinQuant-Expert-14B-v1",
        "adapter_path": str(LEGACY),
        "manifest_path": str(LEGACY_MANIFEST),
        "manifest_sha256": sha256_file(LEGACY_MANIFEST),
        "adapter_sha256": sha256_file(primary),
        "primary_adapter_path": str(primary),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "legacy_read_only": True,
    }
    validate_pointer(pointer)
    return pointer


def promote(manifest_path: Path) -> dict:
    new_pointer = pointer_for_manifest(manifest_path)
    previous = None
    retired_previous = None
    retired_rollback = None
    if CURRENT.exists():
        candidate = read_json(CURRENT)
        try:
            validate_pointer(candidate)
        except ValueError as exc:
            retired_previous = {
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "can_influence_live": False,
                "pointer": candidate,
            }
            retired_path = RETIRED / (
                "incompatible-current-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + f"-{time.time_ns()}"
                + ".json"
            )
            atomic_json(retired_path, retired_previous)
            retired_previous["audit_path"] = str(retired_path)
        else:
            previous = candidate
    if previous and previous.get("adapter_path") == new_pointer.get("adapter_path"):
        previous = None
    if ROLLBACK.exists():
        rollback_candidate = read_json(ROLLBACK)
        try:
            validate_pointer(rollback_candidate)
        except ValueError as exc:
            retired_rollback = {
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "can_influence_live": False,
                "pointer": rollback_candidate,
            }
            retired_path = RETIRED / (
                "incompatible-rollback-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + f"-{time.time_ns()}"
                + ".json"
            )
            atomic_json(retired_path, retired_rollback)
            retired_rollback["audit_path"] = str(retired_path)
            ROLLBACK.unlink()
    if previous and previous.get("adapter_path") != new_pointer.get("adapter_path"):
        atomic_json(ROLLBACK, previous)
    atomic_json(CURRENT, new_pointer)
    return {
        "current": new_pointer,
        "rollback": previous,
        "retired_incompatible_previous": retired_previous,
        "retired_incompatible_rollback": retired_rollback,
    }


def validate_current(*, required: bool = True) -> dict | None:
    if not CURRENT.exists():
        if required:
            raise ValueError("FinQuant current adapter pointer is missing")
        return None
    pointer = read_json(CURRENT)
    validate_pointer(pointer)
    return pointer


def rollback() -> dict:
    current = validate_current(required=True)
    if not ROLLBACK.exists():
        raise ValueError("FinQuant rollback pointer is missing")
    target = read_json(ROLLBACK)
    validate_pointer(target)
    if target.get("adapter_path") == current.get("adapter_path"):
        raise ValueError("FinQuant rollback target equals current adapter")
    timestamp = datetime.now(timezone.utc).isoformat()
    target["updated_at"] = timestamp
    current["updated_at"] = timestamp
    atomic_json(CURRENT, target)
    atomic_json(ROLLBACK, current)
    return {"current": target, "rollback": current}


def status() -> dict:
    current = validate_current(required=True)
    current_manifest, current_path = validate_pointer(current)
    rollback_pointer = read_json(ROLLBACK) if ROLLBACK.exists() else None
    rollback_manifest = None
    rollback_path = None
    if rollback_pointer is not None:
        rollback_manifest, rollback_path = validate_pointer(rollback_pointer)
    return {
        "registry_version": REGISTRY_VERSION,
        "current_verified": True,
        "current": current,
        "current_manifest": current_manifest,
        "current_adapter_path": str(current_path),
        "rollback_present": rollback_pointer is not None,
        "rollback_verified": rollback_pointer is not None,
        "rollback": rollback_pointer,
        "rollback_manifest": rollback_manifest,
        "rollback_adapter_path": str(rollback_path) if rollback_path else None,
    }


def sync_evidence() -> dict:
    pointer = validate_current(required=True)
    manifest, _adapter_path = validate_pointer(pointer)
    legacy = bool(pointer.get("legacy_read_only"))
    verification_status = "verified_legacy_rollback" if legacy else "verified"
    specialization_status = (
        "legacy_rollback_only" if legacy else manifest.get("specialization_status")
    )
    specialization = {
        "verification_status": verification_status,
        "identity_verified": not legacy,
        "legacy_read_only": legacy,
        "adapter_version": pointer.get("adapter_version"),
        "adapter_path": pointer.get("adapter_path"),
        "lora_adapter": pointer.get("adapter_path"),
        "specialization_manifest": pointer.get("manifest_path"),
        "specialization_id": pointer.get("specialization_id"),
        "training_artifact": pointer.get("adapter_path"),
        "manifest_sha256": pointer.get("manifest_sha256"),
        "adapter_sha256": pointer.get("adapter_sha256"),
        "dataset_version": manifest.get("dataset_version"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "dataset_lineage_sha256": manifest.get("dataset_lineage_sha256"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "source_code_version": manifest.get("source_code_version"),
        "source_script_sha256": manifest.get("source_script_sha256"),
        "trainer_code_sha256": manifest.get("trainer_code_sha256"),
        "base_model_repo": manifest.get("base_model_repo"),
        "base_model_config_sha256": manifest.get("base_model_config_sha256"),
        "inference_base_model_config_sha256": manifest.get("inference_base_model_config_sha256"),
        "evaluation_report": manifest.get("evaluation_report"),
        "held_out_eval_loss": manifest.get("held_out_eval_loss"),
        "objective_name": manifest.get("objective_name"),
        "objective_version": manifest.get("objective_version"),
        "preference_contract_version": manifest.get("preference_contract_version"),
        "preference_selection_accuracy": manifest.get("preference_selection_accuracy"),
        "training_stages": manifest.get("training_stages"),
        "trained_at": manifest.get("trained_at"),
        "sample_count": manifest.get("sample_count"),
        "max_steps": manifest.get("max_steps"),
    }
    evidence = {
        "served_model_name": MODEL_NAME,
        "specialization_required": True,
        "specialization_target": MODEL_NAME,
        "specialization_status": specialization_status,
        "base_model_carrier": INFERENCE_BASE,
        **{key: value for key, value in specialization.items() if value is not None},
        "specialization_evidence": {
            key: value for key, value in specialization.items() if value is not None
        },
    }
    updated = []
    for path in (DOWNLOAD_MANIFEST, VALIDATION_MANIFEST):
        data = read_json(path) if path.exists() else {"models": []}
        models = data.setdefault("models", [])
        target = next(
            (
                row
                for row in models
                if isinstance(row, dict)
                and (row.get("slot") == "llm_expert_pool" or row.get("served_model_name") == MODEL_NAME)
            ),
            None,
        )
        if target is None:
            target = {
                "slot": "llm_expert_pool",
                "repo_id": "Qwen/Qwen3-14B-AWQ",
                "path": INFERENCE_BASE,
                "target": INFERENCE_BASE,
                "role": "expert_pool",
                "status": "ready",
                "exists": True,
            }
            models.append(target)
        target.update(evidence)
        data["checked_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, data)
        updated.append(str(path))
    return {"updated": updated, "evidence": evidence, "pointer": pointer}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--manifest", type=Path, required=True)
    subparsers.add_parser("verify")
    subparsers.add_parser("status")
    subparsers.add_parser("rollback")
    subparsers.add_parser("sync-evidence")
    args = parser.parse_args()
    if args.command == "promote":
        result = promote(args.manifest)
    elif args.command == "verify":
        pointer = validate_current(required=True)
        manifest, adapter_path = validate_pointer(pointer)
        result = {"verified": True, "pointer": pointer, "manifest": manifest, "adapter_path": str(adapter_path)}
    elif args.command == "rollback":
        result = rollback()
    elif args.command == "status":
        result = status()
    else:
        result = sync_evidence()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""


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


def _sha256_bytes(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _finalize_dataset_contract(
    dataset_jsonl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    dataset_sha256 = _sha256_bytes(dataset_jsonl)
    source_code_version = _source_code_version()
    source_script_sha256 = _sha256_file(Path(__file__))
    lineage_payload = json.dumps(
        {
            "created_at": manifest.get("created_at"),
            "source": manifest.get("source"),
            "source_transport": manifest.get("source_transport"),
            "source_code_version": source_code_version,
            "source_script_sha256": source_script_sha256,
            "training_repo": REMOTE_TRAIN_BASE_REPO,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lineage_sha256 = _sha256_bytes(lineage_payload)
    dataset_version = f"bb-finquant-sft-v2-{dataset_sha256[:12]}-{lineage_sha256[:8]}"
    finalized = dict(manifest)
    finalized.update(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_policy": "bb_finquant_expert_sft_v2",
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_sha256,
            "dataset_lineage_sha256": lineage_sha256,
            "source_code_version": source_code_version,
            "source_script_sha256": source_script_sha256,
            "base_model_identity": {
                "training_repo": REMOTE_TRAIN_BASE_REPO,
                "training_path": REMOTE_TRAIN_BASE_MODEL,
                "inference_path": REMOTE_INFERENCE_BASE_MODEL,
                "served_base_model": BASE_MODEL_NAME,
            },
            "example_count": sum(1 for line in dataset_jsonl.splitlines() if line.strip()),
        }
    )
    return finalized


def _validate_dataset_contract(dataset_jsonl: str, manifest: dict[str, Any]) -> None:
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported BB-FinQuant dataset schema")
    version = str(manifest.get("dataset_version") or "")
    if not DATASET_VERSION_PATTERN.fullmatch(version):
        raise ValueError("invalid BB-FinQuant dataset version")
    actual_hash = _sha256_bytes(dataset_jsonl)
    if manifest.get("dataset_sha256") != actual_hash:
        raise ValueError("BB-FinQuant dataset SHA-256 mismatch")
    lineage_hash = str(manifest.get("dataset_lineage_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", lineage_hash):
        raise ValueError("BB-FinQuant dataset lineage hash is invalid")
    expected_version = f"bb-finquant-sft-v2-{actual_hash[:12]}-{lineage_hash[:8]}"
    if version != expected_version:
        raise ValueError("BB-FinQuant dataset version does not match its content and lineage")
    expected_count = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
    if int(manifest.get("example_count") or 0) != expected_count:
        raise ValueError("BB-FinQuant dataset example count mismatch")
    if manifest.get("objective_name") != RETURN_OBJECTIVE_NAME:
        raise ValueError("BB-FinQuant dataset return objective mismatch")
    if manifest.get("objective_version") != RETURN_OBJECTIVE_VERSION:
        raise ValueError("BB-FinQuant dataset return objective version mismatch")
    preference_count = 0
    for line_number, line in enumerate(dataset_jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"BB-FinQuant dataset row {line_number} is not valid JSON") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"BB-FinQuant dataset row {line_number} has invalid messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"BB-FinQuant dataset row {line_number} has invalid message")
            if message.get("role") not in {"user", "assistant"}:
                continue
            try:
                content = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} has invalid JSON message content"
                ) from exc
            if not isinstance(content, dict):
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} JSON message must be an object"
                )
        preference = row.get("preference") if isinstance(row, dict) else None
        if preference is not None:
            if not isinstance(preference, dict):
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} has invalid preference"
                )
            if preference.get("contract_version") != PREFERENCE_CONTRACT_VERSION:
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} preference version mismatch"
                )
            if preference.get("objective") != RETURN_OBJECTIVE_NAME:
                raise ValueError(
                    f"BB-FinQuant dataset row {line_number} preference objective mismatch"
                )
            for key in ("prompt", "chosen", "rejected"):
                if not str(preference.get(key) or "").strip():
                    raise ValueError(
                        f"BB-FinQuant dataset row {line_number} preference missing {key}"
                    )
            preference_count += 1
    if preference_count <= 0 or int(manifest.get("preference_example_count") or 0) != preference_count:
        raise ValueError("BB-FinQuant dataset has no valid return-preference examples")
    base_identity = _safe_dict(manifest.get("base_model_identity"))
    if base_identity.get("training_repo") != REMOTE_TRAIN_BASE_REPO:
        raise ValueError("BB-FinQuant training base-model identity mismatch")
    if not str(manifest.get("source_code_version") or "").strip():
        raise ValueError("BB-FinQuant dataset has no source code version")
    script_hash = str(manifest.get("source_script_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", script_hash):
        raise ValueError("BB-FinQuant dataset has no valid source script hash")


def _new_adapter_version(dataset_manifest: dict[str, Any], *, now: datetime | None = None) -> str:
    dataset_hash = str(dataset_manifest.get("dataset_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
        raise ValueError("cannot version adapter without a valid dataset hash")
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{dataset_hash[:12]}"


def _remote_dataset_paths(dataset_version: str) -> tuple[str, str]:
    if not DATASET_VERSION_PATTERN.fullmatch(dataset_version):
        raise ValueError("invalid remote dataset version")
    root = f"{REMOTE_DATASET_VERSIONS_DIR}/{dataset_version}"
    return f"{root}/dataset.jsonl", f"{root}/manifest.json"


def _remote_adapter_paths(adapter_version: str) -> tuple[str, str, str]:
    if not ADAPTER_VERSION_PATTERN.fullmatch(adapter_version):
        raise ValueError("invalid remote adapter version")
    root = f"{REMOTE_ADAPTER_VERSIONS_DIR}/{adapter_version}"
    return (
        root,
        f"{root}/specialization_manifest.json",
        f"{REMOTE_TRAIN_LOG_DIR}/{adapter_version}.log",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "truncated": True,
            "sha256": _sha256_bytes(serialized),
            "type": type(value).__name__,
        }
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {str(key): _bounded_json_value(item, depth=depth + 1) for key, item in items[:24]}
        if len(items) > 24:
            bounded["_truncated_keys"] = len(items) - 24
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_list = [_bounded_json_value(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            bounded_list.append({"_truncated_items": len(value) - 8})
        return bounded_list
    if isinstance(value, str) and len(value) > 240:
        return {
            "truncated": True,
            "sha256": _sha256_bytes(value),
            "preview": value[:160],
            "original_length": len(value),
        }
    return value


def _json_compact(value: Any, limit: int = 2200) -> str:
    original = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(original) <= limit:
        return original
    bounded = _bounded_json_value(value)
    compacted = json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)
    if len(compacted) <= limit:
        return compacted
    fallback = {
        "truncated": True,
        "sha256": _sha256_bytes(original),
        "top_level_keys": sorted(str(key) for key in value) if isinstance(value, dict) else [],
        "preview": compacted[: min(800, max(limit - 300, 0))],
    }
    return json.dumps(fallback, ensure_ascii=False, sort_keys=True)


def _json_object_from_remote_output(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    parsed_span = -1
    text = str(raw or "")
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and end > parsed_span:
            parsed = value
            parsed_span = end
    if parsed is None:
        raise ValueError(f"remote output did not contain a JSON object: {text[:1000]}")
    return parsed


def _trade_response(sample: dict[str, Any]) -> dict[str, Any]:
    realized = _safe_float(sample.get("realized_pnl"))
    fee_signed = _safe_float(sample.get("fee"), -_safe_float(sample.get("fee_estimate")))
    funding_signed = _safe_float(sample.get("funding_fee"))
    liquidation_signed = _safe_float(sample.get("liquidation_penalty"))
    cost_drag = -sum(min(value, 0.0) for value in (fee_signed, funding_signed, liquidation_signed))
    side = str(sample.get("side") or "").lower()
    return {
        "verdict": "good_trade" if realized > 0 else "bad_trade" if realized < 0 else "flat_trade",
        "side": side,
        "net_pnl_after_all_costs_usdt": round(realized, 6),
        "gross_pnl_usdt": round(_safe_float(sample.get("gross_pnl")), 6),
        "fee_usdt_signed": round(fee_signed, 6),
        "funding_fee_usdt_signed": round(funding_signed, 6),
        "liquidation_penalty_usdt_signed": round(liquidation_signed, 6),
        "total_cost_drag_usdt": round(cost_drag, 6),
        "lesson": sample.get("improvement_summary")
        or sample.get("mistake_summary")
        or (
            "Prefer similar setups only when expected net profit, liquidity, and exit discipline are stronger."
            if realized <= 0
            else "This setup had positive after-fee outcome; reuse only with comparable evidence and risk control."
        ),
        "risk_guidance": {
            "increase_size": bool(realized > 0 and _safe_float(sample.get("hold_minutes")) > 0),
            "avoid_tiny_fee_drag": bool(cost_drag > abs(realized) and realized <= 0),
            "requires_after_fee_positive_expectancy": True,
        },
    }


def _shadow_response(sample: dict[str, Any]) -> dict[str, Any]:
    long_return = _safe_float(sample.get("long_return_pct"))
    short_return = _safe_float(sample.get("short_return_pct"))
    best_side = "long" if long_return >= short_return else "short"
    return {
        "verdict": (
            "missed_opportunity" if sample.get("missed_opportunity") else "shadow_observation"
        ),
        "best_side": best_side,
        "long_return_pct": round(long_return, 6),
        "short_return_pct": round(short_return, 6),
        "lesson": "Rank future candidates by after-cost payoff and avoid suppressing high-quality opportunities without explicit counter-evidence.",
        "risk_guidance": {
            "do_not_trade_without_confirmation": True,
            "use_as_shadow_supervision": True,
        },
    }


def _messages(
    kind: str,
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    rejected_response: dict[str, Any] | None = None,
    preference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    row = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_compact(user)},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False, sort_keys=True),
            },
        ],
        "metadata": {
            "kind": kind,
            "symbol": payload.get("symbol"),
            "side": payload.get("side") or payload.get("decision_action"),
            "source_id": payload.get("id"),
        },
    }
    if rejected_response is not None:
        prompt = "\n".join(
            (
                f"<|system|>\n{system}",
                f"<|user|>\n{_json_compact(user)}",
                "<|assistant|>\n",
            )
        )
        row["preference"] = {
            "contract_version": PREFERENCE_CONTRACT_VERSION,
            "objective": RETURN_OBJECTIVE_NAME,
            "prompt": prompt,
            "chosen": json.dumps(response, ensure_ascii=False, sort_keys=True),
            "rejected": json.dumps(rejected_response, ensure_ascii=False, sort_keys=True),
            "metrics": dict(preference_metrics or {}),
        }
    return row


def _trade_rejected_response(response: dict[str, Any]) -> dict[str, Any]:
    rejected = dict(response)
    realized = _safe_float(response.get("net_pnl_after_all_costs_usdt"))
    rejected["verdict"] = "good_trade" if realized <= 0 else "bad_trade"
    rejected["lesson"] = (
        "Repeat this setup and increase risk because outcome frequency matters more than payoff."
        if realized <= 0
        else "Avoid this setup even though its audited fee-after outcome was positive."
    )
    rejected["risk_guidance"] = {
        "increase_size": bool(realized <= 0),
        "avoid_tiny_fee_drag": False,
        "requires_after_fee_positive_expectancy": False,
    }
    return rejected


def _shadow_rejected_response(
    sample: dict[str, Any],
    response: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    long_return = _safe_float(sample.get("long_return_pct"))
    short_return = _safe_float(sample.get("short_return_pct"))
    chosen_side = str(response.get("best_side") or "long")
    rejected_side = "short" if chosen_side == "long" else "long"
    rejected = dict(response)
    rejected["best_side"] = rejected_side
    rejected["lesson"] = "Choose the lower fee-after return side because direction hit rate is enough."
    chosen_return = long_return if chosen_side == "long" else short_return
    rejected_return = short_return if chosen_side == "long" else long_return
    return rejected, {
        "chosen_net_return_after_cost_pct": round(chosen_return, 8),
        "rejected_net_return_after_cost_pct": round(rejected_return, 8),
        "return_uplift_pct": round(chosen_return - rejected_return, 8),
    }


def _return_objective_counterexample() -> dict[str, Any]:
    payload = {
        "scenario": "low_win_high_payoff_vs_high_win_negative_expectancy",
        "candidate_a": {
            "win_rate": 0.35,
            "avg_win_pct": 4.0,
            "avg_loss_pct": -1.0,
            "expected_net_return_after_cost_pct": 0.75,
        },
        "candidate_b": {
            "win_rate": 0.80,
            "avg_win_pct": 0.10,
            "avg_loss_pct": -2.0,
            "expected_net_return_after_cost_pct": -0.32,
        },
    }
    chosen = {
        "verdict": "prefer_candidate_a",
        "best_side": "candidate_a",
        "lesson": "Lower win rate is acceptable when fee-after expectancy and payoff are superior.",
        "risk_guidance": {"requires_after_fee_positive_expectancy": True},
    }
    rejected = {
        "verdict": "prefer_candidate_b",
        "best_side": "candidate_b",
        "lesson": "Prefer the higher win rate despite negative fee-after expectancy.",
        "risk_guidance": {"requires_after_fee_positive_expectancy": False},
    }
    return _messages(
        "return_preference_counterexample",
        payload,
        chosen,
        rejected_response=rejected,
        preference_metrics={
            "chosen_net_return_after_cost_pct": 0.75,
            "rejected_net_return_after_cost_pct": -0.32,
            "return_uplift_pct": 1.07,
        },
    )


async def _load_expert_memory_examples() -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(ExpertMemory)
            .where(ExpertMemory.is_active.is_(True))
            .order_by(ExpertMemory.confidence_score.desc(), ExpertMemory.id.desc())
        )
        rows = list(result.scalars().all())
    examples: list[dict[str, Any]] = []
    for row in rows:
        extra = row.extra if isinstance(row.extra, dict) else {}
        if not (
            extra.get("production_evidence_eligible") is True
            and extra.get("cost_complete") is True
        ):
            continue
        payload = {
            "id": int(row.id or 0),
            "expert_name": row.expert_name,
            "symbol": row.symbol,
            "side": row.side,
            "market_pattern": row.market_pattern,
            "confidence_score": _safe_float(row.confidence_score),
            "success_count": int(row.success_count or 0),
            "failure_count": int(row.failure_count or 0),
            "net_return_after_cost_pct": _safe_float(
                extra.get("net_return_after_cost_pct")
            ),
            "objective": extra.get("objective"),
            "objective_version": extra.get("objective_version"),
        }
        response = {
            "verdict": "expert_memory_lesson",
            "side": row.side,
            "lesson": row.lesson,
            "observation_policy": "fee_after_outcome_only_no_live_risk_authority",
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
            result = await session.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    """))
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


async def build_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    await _assert_training_data_source_ready()
    reflections = await _load_trade_reflection_samples()
    authoritative = await _load_authoritative_trade_samples()
    trade_samples = _merge_trade_samples(reflections, authoritative)
    shadow_samples = await _load_shadow_samples()
    quality = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    quality_report = _safe_dict(quality.get("quality_report"))
    consistency = _safe_dict(quality_report.get("training_label_consistency"))
    if consistency.get("status") == "blocked" or consistency.get("promotion_blocked"):
        raise RuntimeError(
            "BB-FinQuant dataset label consistency is blocked; refusing to export training data: "
            f"{_json_compact(consistency, limit=2400)}"
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
                "notional_usdt",
                "authoritative_pnl_ratio_pct",
                "realized_pnl",
                "gross_pnl",
                "fee",
                "fee_estimate",
                "funding_fee",
                "liquidation_penalty",
                "settlement_components_total",
                "hold_minutes",
                "leverage",
                "outcome",
                "raw_llm_response",
                "settlement_status",
                "settlement_source",
            )
            if key in sample
        }
        response = _trade_response(sample)
        examples.append(
            _messages(
                "closed_trade",
                payload,
                response,
                rejected_response=_trade_rejected_response(response),
                preference_metrics={
                    "chosen_realized_net_pnl_usdt": response.get(
                        "net_pnl_after_all_costs_usdt"
                    ),
                    "chosen_matches_audited_outcome": True,
                    "rejected_matches_audited_outcome": False,
                },
            )
        )
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
        response = _shadow_response(sample)
        rejected, preference_metrics = _shadow_rejected_response(sample, response)
        examples.append(
            _messages(
                "shadow_backtest",
                payload,
                response,
                rejected_response=rejected,
                preference_metrics=preference_metrics,
            )
        )
    examples.extend(await _load_expert_memory_examples())
    examples.append(_return_objective_counterexample())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "example_count": len(examples),
        "example_counts_by_kind": {
            kind: sum(
                1 for example in examples if _safe_dict(example.get("metadata")).get("kind") == kind
            )
            for kind in ("closed_trade", "shadow_backtest", "expert_memory")
        },
        "quality_report": quality_report,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "preference_contract_version": PREFERENCE_CONTRACT_VERSION,
        "preference_example_count": sum(
            1 for example in examples if isinstance(example.get("preference"), dict)
        ),
        "source": "platform_db_clean_training_view",
    }
    return examples, manifest


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    run_remote_text(ssh, f"mkdir -p {sh(posixpath.dirname(remote_path))}", timeout=30)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _upload_text_atomic(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    temporary_path = f"{remote_path}.tmp.{time.time_ns()}"
    _upload_text(ssh, temporary_path, content, mode=mode)
    run_remote_text(
        ssh,
        f"mv -f {sh(temporary_path)} {sh(remote_path)}",
        timeout=30,
        check=True,
    )


def _remote_immutable_file_exists(ssh, remote_path: str, expected_sha256: str) -> bool:
    raw = run_remote_text(
        ssh,
        f"if [ -e {sh(remote_path)} ]; then sha256sum {sh(remote_path)} | cut -d' ' -f1; "
        "else echo missing; fi",
        timeout=30,
        check=True,
    ).strip()
    if raw == "missing":
        return False
    if raw != expected_sha256:
        raise RuntimeError(f"refusing to overwrite immutable remote artifact: {remote_path}")
    return True


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


def export_dataset_from_platform() -> tuple[str, str, dict[str, Any]]:
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
        remote_result: dict[str, Any] = {}
        try:
            remote_result = json.loads(raw)
        except json.JSONDecodeError:
            remote_result = {"raw": raw[:1200]}
        dataset_path = str(remote_result.get("local_dataset") or "")
        manifest_path = str(remote_result.get("local_manifest") or "")
        allowed_prefix = REMOTE_PLATFORM_EXPORT_DIR.rstrip("/") + "/"
        if not dataset_path.startswith(allowed_prefix) or not manifest_path.startswith(
            allowed_prefix
        ):
            raise RuntimeError(
                "online platform export did not return versioned FinQuant artifact paths"
            )
        dataset_jsonl = _download_text(ssh, dataset_path)
        manifest_json = _download_text(ssh, manifest_path)
        return dataset_jsonl, manifest_json, remote_result
    finally:
        ssh.close()


def _remote_service_update_script(*, adapter_path: str) -> str:
    if not str(adapter_path or "").startswith(f"{REMOTE_ADAPTER_ROOT}/"):
        raise ValueError("8003 cannot start without a verified BB-FinQuant adapter")
    lora_args = "  --enable-lora \\\n" f"  --lora-modules {MODEL_NAME}={adapter_path} \\\n"
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
    gateway_script = f"""from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 8003
UPSTREAM = "http://127.0.0.1:8000"
GATEWAY_MODEL = "{MODEL_NAME}"
UPSTREAM_MODEL = "{MODEL_NAME}"


class Handler(BaseHTTPRequestHandler):
    server_version = "BBFinQuantVerifiedGateway/3.0"

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
            try:
                with urllib.request.urlopen(UPSTREAM + "/v1/models", timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                self._write_json(503, {{"error": "finquant_upstream_models_unavailable", "detail": str(exc)[:200]}})
                return
            rows = [
                row for row in payload.get("data", [])
                if isinstance(row, dict) and row.get("id") == GATEWAY_MODEL
            ] if isinstance(payload, dict) else []
            if not rows:
                self._write_json(503, {{"error": "finquant_adapter_not_loaded"}})
                return
            self._write_json(200, {{"object": "list", "data": rows}})
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
            except Exception:
                self._write_json(400, {{"error": "invalid_json_request"}})
                return
            if isinstance(payload, dict) and payload.get("model") != UPSTREAM_MODEL:
                self._write_json(400, {{"error": "finquant_model_identity_required"}})
                return
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
            self._write_json(502, {{"error": "finquant_gateway_upstream_failed", "detail": str(exc)[:200]}})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
"""
    gateway_service = f"""[Unit]
Description=BB FinQuant verified adapter gateway
After=network-online.target qwen3-14b-trade.service
Wants=network-online.target

[Service]
Type=simple
User=linux
WorkingDirectory={REMOTE_GATEWAY_DIR}
ExecStart=/usr/bin/python3 {REMOTE_GATEWAY_SCRIPT}
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
"""
    gateway_service_upload = f"{REMOTE_SERVICE_DIR}/{REMOTE_GATEWAY_SERVICE}"
    return textwrap.dedent(f"""
        set -euo pipefail
        cp {sh(REMOTE_QWEN_START_SCRIPT)} {sh(REMOTE_QWEN_START_SCRIPT + '.bak.' + str(int(time.time())))}
        cat > {sh(REMOTE_QWEN_START_SCRIPT)} <<'SH'
{qwen_script.rstrip()}
SH
        chmod +x {sh(REMOTE_QWEN_START_SCRIPT)}
        mkdir -p {sh(REMOTE_GATEWAY_DIR)} {sh(REMOTE_SERVICE_DIR)}
        cat > {sh(REMOTE_GATEWAY_SCRIPT)} <<'PY'
{gateway_script.rstrip()}
PY
        chmod +x {sh(REMOTE_GATEWAY_SCRIPT)}
        cat > {sh(gateway_service_upload)} <<'UNIT'
{gateway_service.rstrip()}
UNIT
        sudo -n install -m 0644 {sh(gateway_service_upload)} {sh(REMOTE_GATEWAY_SERVICE_PATH)}
        sudo -n systemctl disable --now {sh(REMOTE_LEGACY_ALIAS_SERVICE)} || true
        sudo -n systemctl daemon-reload
        sudo -n systemctl restart qwen3-14b-trade.service
        sudo -n systemctl enable --now {sh(REMOTE_GATEWAY_SERVICE)}
        sudo -n systemctl restart {sh(REMOTE_GATEWAY_SERVICE)}
        """).strip()


def _switch_verified_adapter(ssh, *, rollback_service: bool) -> dict[str, Any]:
    _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
    result: dict[str, Any] = {}
    if rollback_service:
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} rollback",
            timeout=120,
            check=True,
        )
        result["rollback"] = _json_object_from_remote_output(raw)
    verified = run_remote_text(
        ssh,
        f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} verify",
        timeout=180,
        check=True,
    )
    verified_state = _json_object_from_remote_output(verified)
    adapter = str(verified_state.get("adapter_path") or "")
    if not adapter:
        raise RuntimeError("verified FinQuant adapter state did not include an adapter path")
    run_remote_text(
        ssh,
        _remote_service_update_script(adapter_path=adapter),
        timeout=180,
        check=True,
    )
    inference_request = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": 'Return compact JSON: {"status":"ok"}.'}],
            "temperature": 0,
            "max_tokens": 16,
        },
        separators=(",", ":"),
    )
    probe = run_remote_text(
        ssh,
        "set +e; "
        "gateway_payload=''; "
        "upstream_payload=''; "
        "for i in $(seq 1 120); do "
        "  upstream_payload=$(curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models 2>&1); "
        "  rc=$?; "
        '  if [ "$rc" -eq 0 ]; then break; fi; '
        "  sleep 3; "
        "done; "
        "gateway_payload=$(curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models 2>&1); "
        f"inference_payload=$(curl -fsS --max-time 180 -H 'content-type: application/json' -d {sh(inference_request)} http://127.0.0.1:8003/v1/chat/completions 2>&1); "
        f"process_payload=$(ps -eo args | grep -F -- {sh('--lora-modules ' + MODEL_NAME + '=' + adapter)} | grep -v grep | head -1); "
        'printf \'%s\\n---8000---\\n%s\\n---INFERENCE---\\n%s\\n---PROCESS---\\n%s\\n\' "$gateway_payload" "$upstream_payload" "$inference_payload" "$process_payload"',
        timeout=600,
        check=False,
    )
    gateway_probe, _separator, remainder = probe.partition("---8000---")
    upstream_probe, _separator, remainder = remainder.partition("---INFERENCE---")
    inference_probe, _separator, process_probe = remainder.partition("---PROCESS---")
    if MODEL_NAME not in gateway_probe or BASE_MODEL_NAME not in upstream_probe:
        raise RuntimeError(
            "BB-FinQuant service switch did not verify both gateway and upstream "
            f"models. Probe excerpt:\n{probe[:5000]}"
        )
    if MODEL_NAME not in upstream_probe:
        raise RuntimeError(
            "BB-FinQuant adapter was not exposed by upstream vLLM. "
            f"Probe excerpt:\n{probe[:5000]}"
        )
    if '"choices"' not in inference_probe or adapter not in process_probe:
        raise RuntimeError(
            "BB-FinQuant service did not prove real adapter inference and process identity. "
            f"Probe excerpt:\n{probe[:5000]}"
        )
    synced = run_remote_text(
        ssh,
        f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} sync-evidence",
        timeout=180,
        check=True,
    )
    result.update(
        {
            "service_switched": True,
            "adapter_state": verified_state,
            "probe": probe[:5000],
            "specialization_evidence": _json_object_from_remote_output(synced),
        }
    )
    return result


def rollback_and_switch_service() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        return _switch_verified_adapter(ssh, rollback_service=True)
    finally:
        ssh.close()


def remote_registry_status() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} status",
            timeout=180,
            check=True,
        )
        return _json_object_from_remote_output(raw)
    finally:
        ssh.close()


def sync_remote_registry_evidence() -> dict[str, Any]:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        raw = run_remote_text(
            ssh,
            f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} sync-evidence",
            timeout=180,
            check=True,
        )
        return _json_object_from_remote_output(raw)
    finally:
        ssh.close()


def deploy_and_optionally_train(
    *,
    dataset_jsonl: str,
    manifest_json: str,
    train: bool,
    switch_service: bool,
    stop_inference_for_training: bool,
    max_steps: int,
    adapter_version: str | None = None,
) -> dict[str, Any]:
    try:
        dataset_manifest = json.loads(manifest_json)
    except json.JSONDecodeError as exc:
        raise ValueError("BB-FinQuant dataset manifest is not valid JSON") from exc
    if not isinstance(dataset_manifest, dict):
        raise ValueError("BB-FinQuant dataset manifest must be an object")
    _validate_dataset_contract(dataset_jsonl, dataset_manifest)
    dataset_version = str(dataset_manifest["dataset_version"])
    remote_dataset, remote_dataset_manifest = _remote_dataset_paths(dataset_version)
    if train and not stop_inference_for_training:
        raise ValueError("LoRA training requires stopping all conflicting 14B services")
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        if not _remote_immutable_file_exists(
            ssh,
            remote_dataset,
            str(dataset_manifest["dataset_sha256"]),
        ):
            _upload_text_atomic(ssh, remote_dataset, dataset_jsonl)
        if not _remote_immutable_file_exists(
            ssh,
            remote_dataset_manifest,
            _sha256_bytes(manifest_json),
        ):
            _upload_text_atomic(ssh, remote_dataset_manifest, manifest_json)
        _upload_text_atomic(
            ssh,
            REMOTE_DATASET_CURRENT,
            json.dumps(
                {
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "dataset_version": dataset_version,
                    "dataset_sha256": dataset_manifest["dataset_sha256"],
                    "dataset_path": remote_dataset,
                    "manifest_path": remote_dataset_manifest,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _upload_text(ssh, REMOTE_TRAINER, REMOTE_TRAINER_CODE, mode=0o755)
        _upload_text(ssh, REMOTE_REGISTRY_TOOL, REMOTE_REGISTRY_TOOL_CODE, mode=0o755)
        result: dict[str, Any] = {
            "uploaded": {
                "dataset": remote_dataset,
                "dataset_manifest": remote_dataset_manifest,
                "dataset_pointer": REMOTE_DATASET_CURRENT,
                "trainer": REMOTE_TRAINER,
                "registry_tool": REMOTE_REGISTRY_TOOL,
            },
            "trained": False,
            "service_switched": False,
        }
        if train:
            selected_version = adapter_version or _new_adapter_version(dataset_manifest)
            adapter_dir, specialization_manifest, train_log = _remote_adapter_paths(
                selected_version
            )
            pre = (
                f"sudo -n systemctl stop {REMOTE_GATEWAY_SERVICE} "
                f"{REMOTE_LEGACY_ALIAS_SERVICE} qwen3-14b-trade.service "
                "deepseek-r1-14b-risk.service || true; "
            )
            post = (
                "sudo -n systemctl start qwen3-14b-trade.service "
                f"{REMOTE_GATEWAY_SERVICE} deepseek-r1-14b-risk.service || true; "
            )
            train_cmd = (
                "set -euo pipefail; "
                f"mkdir -p {sh(REMOTE_TRAINING_DIR)} {sh(REMOTE_ADAPTER_VERSIONS_DIR)} "
                f"{sh(REMOTE_TRAIN_LOG_DIR)}; "
                f"{_remote_train_base_prepare_command()}; "
                "/data/BB/envs/phase3-quant/bin/python -c "
                "'import datasets, trl; print(trl.__version__)'; "
                f"{pre}"
                f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_TRAINER)} "
                f"--dataset {sh(remote_dataset)} "
                f"--dataset-manifest {sh(remote_dataset_manifest)} "
                f"--output-dir {sh(adapter_dir)} "
                f"--base-model {sh(REMOTE_TRAIN_BASE_MODEL)} "
                f"--base-model-repo {sh(REMOTE_TRAIN_BASE_REPO)} "
                f"--inference-base-model {sh(REMOTE_INFERENCE_BASE_MODEL)} "
                f"--manifest {sh(specialization_manifest)} "
                f"--version-id {sh(selected_version)} "
                f"--max-steps {int(max_steps)} "
                f"> {sh(train_log)} 2>&1; "
                f"{post}"
                f"cat {sh(specialization_manifest)}"
            )
            try:
                raw = run_remote_text(ssh, train_cmd, timeout=7200, check=True)
            except Exception as exc:
                tail = run_remote_text(
                    ssh,
                    f"tail -n 160 {sh(train_log)} 2>/dev/null || true; "
                    "sudo -n systemctl start qwen3-14b-trade.service "
                    f"{REMOTE_GATEWAY_SERVICE} deepseek-r1-14b-risk.service || true",
                    timeout=120,
                    check=False,
                )
                raise RuntimeError(
                    "remote LoRA training failed. "
                    f"Command error: {safe_error_text(exc, limit=1400)}\n"
                    f"Train log tail:\n{tail}"
                ) from None
            result["trained"] = True
            result["adapter_version"] = selected_version
            result["specialization_manifest"] = _json_object_from_remote_output(raw)
            promoted = run_remote_text(
                ssh,
                f"/data/BB/envs/phase3-quant/bin/python {sh(REMOTE_REGISTRY_TOOL)} "
                f"promote --manifest {sh(specialization_manifest)}",
                timeout=180,
                check=True,
            )
            result["artifact_registry"] = _json_object_from_remote_output(promoted)
        if switch_service:
            result.update(_switch_verified_adapter(ssh, rollback_service=False))
        return result
    finally:
        ssh.close()


def _remote_train_base_prepare_command() -> str:
    return textwrap.dedent(f"""
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
        """).strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "platform"),
        default="auto",
        help=(
            "Training data source. auto uses the local DB when it has the required "
            "training tables, otherwise exports from the online platform server."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Reuse one already-exported immutable local dataset version.",
    )
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--switch-service", action="store_true")
    parser.add_argument("--rollback-service", action="store_true")
    parser.add_argument("--registry-status", action="store_true")
    parser.add_argument("--sync-registry-evidence", action="store_true")
    parser.add_argument("--stop-inference-for-training", action="store_true")
    args = parser.parse_args()

    if args.registry_status:
        if (
            args.train
            or args.switch_service
            or args.rollback_service
            or args.export_only
            or args.sync_registry_evidence
        ):
            raise SystemExit("--registry-status cannot be combined with mutating operations")
        result = remote_registry_status()
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.sync_registry_evidence:
        if args.train or args.switch_service or args.rollback_service or args.export_only:
            raise SystemExit("--sync-registry-evidence cannot be combined with other operations")
        result = sync_remote_registry_evidence()
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.rollback_service:
        if not args.switch_service:
            raise SystemExit("--rollback-service requires --switch-service")
        if args.train or args.export_only:
            raise SystemExit("rollback cannot be combined with training or dataset export")
        result = rollback_and_switch_service()
        result["source"] = "existing_verified_remote_artifact"
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    requested_dataset_version = str(args.dataset_version or "").strip()
    if requested_dataset_version:
        if not DATASET_VERSION_PATTERN.fullmatch(requested_dataset_version):
            raise SystemExit("invalid --dataset-version")
        source = "immutable_local_version"
        local_dir = ROOT / "data" / "finquant_expert_training"
        version_dir = local_dir / "versions" / requested_dataset_version
        local_dataset = version_dir / "dataset.jsonl"
        local_manifest = version_dir / "manifest.json"
        if not local_dataset.is_file() or not local_manifest.is_file():
            raise SystemExit(f"dataset version is incomplete or missing: {version_dir}")
        dataset_jsonl = local_dataset.read_bytes().decode("utf-8")
        try:
            manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"dataset manifest is invalid: {local_manifest}") from exc
        _validate_dataset_contract(dataset_jsonl, manifest)
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        example_count = int(manifest["example_count"])
    else:
        source = args.source
        if source == "auto":
            try:
                local_ready, _missing = await _training_data_source_ready()
            except SQLAlchemyError:
                local_ready = False
            source = "local" if local_ready else "platform"

    if not requested_dataset_version and source == "platform":
        dataset_jsonl, manifest_json, platform_export = export_dataset_from_platform()
        try:
            manifest = json.loads(manifest_json)
        except json.JSONDecodeError:
            manifest = {}
        example_count = int(manifest.get("example_count") or 0)
        if example_count <= 0:
            example_count = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
        manifest["source_transport"] = "online_platform_export"
        manifest["platform_export"] = platform_export
    elif not requested_dataset_version:
        examples, manifest = await build_dataset()
        if not examples:
            raise SystemExit("No BB-FinQuant SFT examples were generated.")
        dataset_jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
        manifest["source_transport"] = "local_database"
        example_count = len(examples)

    if not dataset_jsonl.strip():
        raise SystemExit("No BB-FinQuant SFT examples were generated.")
    if not requested_dataset_version:
        manifest = _finalize_dataset_contract(dataset_jsonl, manifest)
    _validate_dataset_contract(dataset_jsonl, manifest)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    example_count = int(manifest["example_count"])
    local_dir = ROOT / "data" / "finquant_expert_training"
    version_dir = local_dir / "versions" / str(manifest["dataset_version"])
    local_dataset = version_dir / "dataset.jsonl"
    local_manifest = version_dir / "manifest.json"
    if version_dir.exists():
        if not local_dataset.is_file() or not local_manifest.is_file():
            raise RuntimeError(f"incomplete immutable dataset version: {version_dir}")
        if (
            _sha256_file(local_dataset) != manifest["dataset_sha256"]
            or local_manifest.read_text(encoding="utf-8") != manifest_json
        ):
            raise RuntimeError(f"refusing to overwrite immutable dataset version: {version_dir}")
    else:
        version_dir.mkdir(parents=True, exist_ok=False)
        local_dataset.write_bytes(dataset_jsonl.encode("utf-8"))
        local_manifest.write_text(manifest_json, encoding="utf-8")
    _write_json_atomic(
        local_dir / "current.json",
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_version": manifest["dataset_version"],
            "dataset_sha256": manifest["dataset_sha256"],
            "dataset_path": str(local_dataset.relative_to(local_dir)),
            "manifest_path": str(local_manifest.relative_to(local_dir)),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    if args.export_only:
        safe_print(
            json.dumps(
                {
                    "exported": True,
                    "source": source,
                    "dataset_version": manifest["dataset_version"],
                    "dataset_sha256": manifest["dataset_sha256"],
                    "local_dataset": str(local_dataset),
                    "local_manifest": str(local_manifest),
                    "example_count": example_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = deploy_and_optionally_train(
        dataset_jsonl=dataset_jsonl,
        manifest_json=manifest_json,
        train=args.train,
        switch_service=args.switch_service,
        stop_inference_for_training=args.stop_inference_for_training,
        max_steps=args.max_steps,
    )
    result["source"] = source
    result["example_count"] = example_count
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
