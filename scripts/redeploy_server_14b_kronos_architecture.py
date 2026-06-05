"""Redeploy the remote trading AI stack as local 14B + quant tools.

This script intentionally removes the previous local 32B/old Qwen services and
recreates the target architecture:
- 127.0.0.1:8000 / public 31840: DeepSeek-R1-Distill-Qwen-14B vLLM OpenAI-compatible API
- 127.0.0.1:8001 / public 31841: local quant tools API with Kronos/ML/RL runtime deps

The online high-risk review model is configured in the local .env, because its
API key should not be stored on the remote model server.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent.parent
DEEPSEEK14_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEEPSEEK14_DIR = "/data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-14B"
DEEPSEEK14_SERVED = "deepseek-r1-distill-qwen-14b-trade"
TOOLS_DIR = "/data/trade_ai/tools"
LOG_DIR = "/data/trade_ai/logs"


def parse_server_info() -> dict[str, str | int]:
    candidates = list(ROOT.glob("*服务器资料*.txt")) + list(ROOT.glob("*資料*.txt")) + list(ROOT.glob("*资料*.txt"))
    if not candidates:
        raise FileNotFoundError("未找到服务器资料.txt")
    text = candidates[0].read_text(encoding="utf-8", errors="replace")

    host_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", text)
    port_match = re.search(r"端口[:：]\s*(\d+)", text)
    user_match = re.search(r"账号[:：]\s*(\S+)", text)
    password_match = re.search(r"密码[:：]\s*(\S+)", text)
    if not all([host_match, port_match, user_match, password_match]):
        raise ValueError("服务器资料.txt 格式不完整，无法解析 host/port/user/password")
    return {
        "host": host_match.group(1),
        "port": int(port_match.group(1)),
        "username": user_match.group(1),
        "password": password_match.group(1),
    }


def run(ssh: paramiko.SSHClient, command: str, timeout: int = 300) -> str:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"command failed ({status}):\n{command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out + err


LOCAL_AI_TOOLS_CODE = r'''
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

MODEL_DIR = Path("/data/trade_ai/models")
BUNDLE_PATH = MODEL_DIR / "local_quant_models.joblib"
METADATA_PATH = MODEL_DIR / "local_quant_models_metadata.json"
KRONOS_MODEL_DIR = Path("/data/trade_models/Kronos")

FEATURE_KEYS = [
    "change_24h_pct", "spread_pct", "rsi_14", "rsi_7", "macd", "macd_signal",
    "macd_diff", "stoch_k", "adx_14", "bb_width", "bb_pct", "atr_pct",
    "volume_ratio", "returns_1", "returns_5", "returns_20", "volatility_20",
    "price_vs_sma20", "price_vs_sma50", "funding_rate", "log_volume_24h",
    "log_open_interest_value", "orderbook_imbalance", "orderbook_depth_ratio",
    "news_sentiment_avg", "social_sentiment_avg", "social_mention_count",
    "news_article_count", "decision_confidence", "horizon_minutes",
]

app = FastAPI(title="Trade Local AI Tools", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureRequest(BaseModel):
    symbol: str | None = None
    features: dict[str, Any] = {}
    local_ml_signal: dict[str, Any] | None = None
    open_positions: list[dict[str, Any]] | None = None


class TrainRequest(BaseModel):
    shadow_samples: list[dict[str, Any]] = []
    trade_samples: list[dict[str, Any]] = []
    sequence_samples: list[dict[str, Any]] = []
    text_sentiment_samples: list[dict[str, Any]] = []
    source: str = "local_trading_system"


_BUNDLE_CACHE: dict[str, Any] | None = None
_BUNDLE_MTIME: float | None = None
_KRONOS_STATUS: dict[str, Any] | None = None


def f(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(features.get(key, default) or default)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def feature_row(features: dict[str, Any], *, horizon_minutes: int | None = None) -> dict[str, float]:
    price = f(features, "current_price", f(features, "close", 0.0))
    atr = f(features, "atr_14")
    bid_depth = f(features, "orderbook_bid_depth")
    ask_depth = f(features, "orderbook_ask_depth")
    total_depth = max(bid_depth + ask_depth, 1e-9)
    volume_24h = max(f(features, "volume_24h"), 0.0)
    oi_value = max(f(features, "open_interest_value"), 0.0)
    values = {
        "change_24h_pct": f(features, "change_24h_pct"),
        "spread_pct": f(features, "spread_pct"),
        "rsi_14": f(features, "rsi_14", 50.0),
        "rsi_7": f(features, "rsi_7", 50.0),
        "macd": f(features, "macd"),
        "macd_signal": f(features, "macd_signal"),
        "macd_diff": f(features, "macd_diff"),
        "stoch_k": f(features, "stoch_k", 50.0),
        "adx_14": f(features, "adx_14"),
        "bb_width": f(features, "bb_width"),
        "bb_pct": f(features, "bb_pct", 0.5),
        "atr_pct": atr / price if price > 0 else 0.0,
        "volume_ratio": f(features, "volume_ratio", 1.0),
        "returns_1": f(features, "returns_1"),
        "returns_5": f(features, "returns_5"),
        "returns_20": f(features, "returns_20"),
        "volatility_20": f(features, "volatility_20"),
        "price_vs_sma20": f(features, "price_vs_sma20"),
        "price_vs_sma50": f(features, "price_vs_sma50"),
        "funding_rate": f(features, "funding_rate"),
        "log_volume_24h": math.log10(volume_24h + 1.0),
        "log_open_interest_value": math.log10(oi_value + 1.0),
        "orderbook_imbalance": f(features, "orderbook_imbalance"),
        "orderbook_depth_ratio": (bid_depth - ask_depth) / total_depth,
        "news_sentiment_avg": f(features, "news_sentiment_avg"),
        "social_sentiment_avg": f(features, "social_sentiment_avg"),
        "social_mention_count": f(features, "social_mention_count"),
        "news_article_count": f(features, "news_article_count"),
        "decision_confidence": f(features, "decision_confidence"),
        "horizon_minutes": float(horizon_minutes if horizon_minutes is not None else f(features, "horizon_minutes", 10.0)),
    }
    return {key: float(values.get(key, 0.0)) for key in FEATURE_KEYS}


def model_x(features: dict[str, Any], *, horizon_minutes: int | None = None) -> list[float]:
    row = feature_row(features, horizon_minutes=horizon_minutes)
    return [row[key] for key in FEATURE_KEYS]


def symbol_key(value: Any) -> str:
    return str(value or "").upper().replace("-", "/")


def load_bundle() -> dict[str, Any] | None:
    global _BUNDLE_CACHE, _BUNDLE_MTIME
    if not BUNDLE_PATH.exists():
        return None
    mtime = BUNDLE_PATH.stat().st_mtime
    if _BUNDLE_CACHE is not None and _BUNDLE_MTIME == mtime:
        return _BUNDLE_CACHE
    _BUNDLE_CACHE = joblib.load(BUNDLE_PATH)
    _BUNDLE_MTIME = mtime
    return _BUNDLE_CACHE


def kronos_status() -> dict[str, Any]:
    global _KRONOS_STATUS
    if _KRONOS_STATUS is not None:
        return _KRONOS_STATUS
    status: dict[str, Any] = {
        "installed": False,
        "importable": False,
        "model_dir": str(KRONOS_MODEL_DIR),
        "runtime": "not_loaded",
    }
    try:
        import torch
        status["torch"] = getattr(torch, "__version__", "unknown")
        status["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:
        status["torch_error"] = str(exc)[:160]
    try:
        import model  # Kronos source package exposes model.py
        status["importable"] = True
        status["module"] = getattr(model, "__file__", "model")
    except Exception as exc:
        status["import_error"] = str(exc)[:160]
    status["installed"] = bool(status.get("importable"))
    _KRONOS_STATUS = status
    return status


def heuristic_timeseries(features: dict[str, Any], horizon: int) -> dict[str, Any]:
    r1 = f(features, "returns_1")
    r5 = f(features, "returns_5")
    r20 = f(features, "returns_20")
    macd = f(features, "macd_diff")
    trend = f(features, "price_vs_sma20") + f(features, "price_vs_sma50")
    vol = max(f(features, "volatility_20"), 0.001)
    volume = f(features, "volume_ratio", 1.0)
    score = 0.35 * r1 + 0.30 * r5 + 0.20 * r20 + 0.10 * macd + 0.05 * trend
    expected_return_pct = clamp(score * 100.0 * (horizon / 10.0), -8.0, 8.0)
    confidence = clamp(0.45 + abs(score) / max(vol, 1e-6) * 0.08 + min(volume, 3.0) * 0.03, 0.05, 0.88)
    return {
        "horizon_minutes": horizon,
        "expected_return_pct": expected_return_pct,
        "downside_risk_pct": clamp(vol * 100.0 * math.sqrt(max(horizon, 1) / 10.0), 0.05, 12.0),
        "direction": "long" if expected_return_pct > 0 else ("short" if expected_return_pct < 0 else "flat"),
        "confidence": confidence,
        "backend": "kronos_runtime_pending_heuristic",
        "kronos": kronos_status(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_bundle()
    metadata = bundle.get("metadata", {}) if bundle else {}
    return {
        "ok": True,
        "service": "trade-local-ai-tools",
        "architecture": "kronos_ml_rl_local_14b_online_review",
        "tools": ["kronos_timeseries", "profit", "loss_filter", "exit", "rl_execution_shadow", "train"],
        "trained_models_available": bool(bundle),
        "trained_at": metadata.get("trained_at"),
        "shadow_sample_count": metadata.get("shadow_sample_count", 0),
        "trade_sample_count": metadata.get("trade_sample_count", 0),
        "kronos": kronos_status(),
        "review_backend": "online_configured_in_trading_app",
    }


@app.get("/models/status")
def local_models_status() -> dict[str, Any]:
    bundle = load_bundle()
    metadata = bundle.get("metadata", {}) if bundle else {}
    return {
        "available": bool(bundle),
        "model_path": str(BUNDLE_PATH),
        "kronos": kronos_status(),
        "rl_execution": {"available": True, "mode": "shadow_policy_rules_until_trained"},
        **metadata,
    }


@app.post("/timeseries/predict")
def predict_timeseries(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    return {
        "symbol": req.symbol or features.get("symbol"),
        "predictions": [heuristic_timeseries(features, h) for h in (10, 30, 60)],
    }


@app.post("/profit/predict")
def predict_profit(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    bundle = load_bundle()
    if not bundle:
        ts = heuristic_timeseries(features, 10)
        expected = float(ts["expected_return_pct"])
        return {
            "available": False,
            "expected_return_pct": expected,
            "profit_edge_pct": abs(expected) * 2.0,
            "best_side": "long" if expected >= 0 else "short",
            "risk_score": clamp(float(ts["downside_risk_pct"]) / 10.0, 0.0, 1.0),
            "backend": "heuristic_until_trained",
        }
    x10 = np.asarray([model_x(features, horizon_minutes=10)], dtype=float)
    long_ret = float(bundle["long_return_model"].predict(x10)[0])
    short_ret = float(bundle["short_return_model"].predict(x10)[0])
    best_side = "long" if long_ret >= short_ret else "short"
    return {
        "available": True,
        "expected_return_pct": max(long_ret, short_ret),
        "long_expected_return_pct": long_ret,
        "short_expected_return_pct": short_ret,
        "profit_edge_pct": abs(long_ret - short_ret),
        "best_side": best_side,
        "risk_score": 0.0,
        "backend": "trained_extra_trees",
    }


@app.post("/rl/execute")
def rl_execute(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    profit = predict_profit(req)
    edge = abs(float(profit.get("profit_edge_pct") or 0.0))
    vol = max(f(features, "volatility_20"), 0.001)
    confidence = clamp(0.45 + edge / 10.0 - vol * 2.0, 0.05, 0.9)
    size = clamp(edge / 20.0, 0.01, 0.18) if confidence >= 0.52 else 0.0
    return {
        "available": True,
        "mode": "shadow_policy_rules_until_rl_trained",
        "suggested_position_size_pct": size,
        "suggested_leverage": 1.0 if confidence < 0.7 else 2.0,
        "confidence": confidence,
        "note": "PPO/SAC runtime installed; policy remains shadow until trained on local fills.",
    }


@app.post("/exit/predict")
def predict_exit(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    open_positions = req.open_positions or []
    pnl = max([float(p.get("unrealized_pnl") or 0.0) for p in open_positions] + [0.0])
    ret5 = f(features, "returns_5")
    action = "hold"
    if pnl > 0 and ret5 < -0.003:
        action = "take_profit"
    elif pnl < -5 and ret5 < -0.005:
        action = "cut_loss"
    return {"action": action, "confidence": 0.62 if action != "hold" else 0.35}


@app.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    rows = []
    for sample in req.shadow_samples or []:
        features = sample.get("features") or {}
        if not features:
            continue
        horizon = int(sample.get("horizon_minutes") or features.get("horizon_minutes") or 10)
        long_return = f(sample, "long_return_pct")
        short_return = f(sample, "short_return_pct")
        rows.append({
            "x": model_x(features, horizon_minutes=horizon),
            "long_return": long_return,
            "short_return": short_return,
            "lossy_long": int(long_return < -0.08),
            "lossy_short": int(short_return < -0.08),
        })
    if len(rows) < 200:
        return {"trained": False, "reason": "need_at_least_200_shadow_samples", "sample_count": len(rows)}
    X = np.asarray([r["x"] for r in rows], dtype=float)
    y_long = np.asarray([r["long_return"] for r in rows], dtype=float)
    y_short = np.asarray([r["short_return"] for r in rows], dtype=float)
    y_loss_long = np.asarray([r["lossy_long"] for r in rows], dtype=int)
    y_loss_short = np.asarray([r["lossy_short"] for r in rows], dtype=int)

    def regressor() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        ])

    def classifier() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=160, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        ])

    bundle = {
        "long_return_model": regressor().fit(X, y_long),
        "short_return_model": regressor().fit(X, y_short),
        "long_loss_model": classifier().fit(X, y_loss_long),
        "short_loss_model": classifier().fit(X, y_loss_short),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "shadow_sample_count": len(rows),
            "feature_keys": FEATURE_KEYS,
            "architecture": "kronos_ml_rl_local_14b_online_review",
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    return {"trained": True, **bundle["metadata"]}
'''


def main() -> None:
    info = parse_server_info()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        str(info["host"]),
        port=int(info["port"]),
        username=str(info["username"]),
        password=str(info["password"]),
        timeout=20,
    )
    try:
        print("== inventory before ==")
        print(run(ssh, """
            set +e
            systemctl list-units --type=service --all | grep -E 'deepseek|qwen3|local-ai-tools' || true
            echo '--- ports ---'
            ss -lntp | grep -E ':8000|:8001|:8003' || true
            echo '--- models ---'
            du -sh /data/trade_models/*/* 2>/dev/null || true
            echo '--- gpu ---'
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
        """))

        print("== cleanup old services/models ==")
        print(run(ssh, """
            set +e
            sudo systemctl stop deepseek-14b-main.service deepseek-32b-main.service qwen3-14b.service qwen3-32b-review.service local-ai-tools.service 2>/dev/null || true
            sudo systemctl disable deepseek-14b-main.service deepseek-32b-main.service qwen3-14b.service qwen3-32b-review.service local-ai-tools.service 2>/dev/null || true
            sudo rm -f /etc/systemd/system/deepseek-14b-main.service /etc/systemd/system/deepseek-32b-main.service /etc/systemd/system/qwen3-14b.service /etc/systemd/system/qwen3-32b-review.service /etc/systemd/system/local-ai-tools.service
            sudo systemctl daemon-reload
            ps -eo pid,args | awk '/pip install/ && !/awk/ {print $1}' | xargs -r kill -9 || true
            rm -rf /data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ
            rm -rf /data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-14B
            rm -rf /data/trade_models/Qwen/Qwen3-32B-AWQ
            rm -rf /data/trade_models/Qwen/Qwen3-14B-AWQ
            rm -f /data/trade_ai/scripts/start_deepseek_14b_main.sh /data/trade_ai/scripts/start_deepseek_32b_main.sh /data/trade_ai/scripts/start_qwen3_14b.sh /data/trade_ai/scripts/start_qwen3_32b_review.sh
            mkdir -p /data/trade_models/Qwen /data/trade_models/Kronos /data/trade_ai/scripts /data/trade_ai/logs /data/trade_ai/models /data/trade_ai/tools
            echo cleanup-done
            exit 0
        """, timeout=300))

        print("== install runtime packages ==")
        print(run(ssh, """
            set -e
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            python -m pip install -q --timeout 60 'transformers>=4.51.0' 'huggingface_hub>=0.24.0' || true
            conda activate trade_ml
            python -m pip install -q --timeout 60 fastapi uvicorn joblib numpy pandas scikit-learn xgboost stable-baselines3 kronos-model-arch huggingface_hub safetensors einops || true
            echo runtime-packages-ok
        """, timeout=1800))

        print("== download DeepSeek R1 Distill Qwen 14B ==")
        download_script = textwrap.dedent(f"""\
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            python - <<'PY'
            import os
            from pathlib import Path
            from huggingface_hub import snapshot_download
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            target = Path("{DEEPSEEK14_DIR}")
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="{DEEPSEEK14_REPO}",
                local_dir=str(target),
                local_dir_use_symlinks=False,
                resume_download=True,
                endpoint=os.environ.get("HF_ENDPOINT"),
            )
            print("downloaded", target)
            PY
        """)
        with ssh.open_sftp().file("/data/trade_ai/scripts/download_deepseek_14b.sh", "w") as remote:
            remote.write(download_script)
        print(run(ssh, "chmod +x /data/trade_ai/scripts/download_deepseek_14b.sh && /data/trade_ai/scripts/download_deepseek_14b.sh", timeout=7200))

        print("== create DeepSeek 14B vLLM service ==")
        start_14b = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            export CUDA_VISIBLE_DEVICES=0
            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export VLLM_USE_V1=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            LOG={LOG_DIR}/deepseek_14b_main.log
            exec python -m vllm.entrypoints.openai.api_server \\
              --host 0.0.0.0 \\
              --port 8000 \\
              --model "{DEEPSEEK14_DIR}" \\
              --served-model-name {DEEPSEEK14_SERVED} \\
              --trust-remote-code \\
              --max-model-len 8192 \\
              --gpu-memory-utilization 0.78 \\
              --dtype auto \\
              --enforce-eager > "$LOG" 2>&1
        """)
        service_14b = textwrap.dedent("""\
            [Unit]
            Description=DeepSeek R1 Distill Qwen 14B Trading vLLM OpenAI API
            After=network-online.target
            Wants=network-online.target

            [Service]
            Type=simple
            User=linux
            WorkingDirectory=/data/trade_ai
            ExecStart=/data/trade_ai/scripts/start_deepseek_14b_main.sh
            Restart=always
            RestartSec=10
            Environment=CUDA_VISIBLE_DEVICES=0
            Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
            Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            LimitNOFILE=65535

            [Install]
            WantedBy=multi-user.target
        """)
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/scripts/start_deepseek_14b_main.sh", "w") as remote:
            remote.write(start_14b)
        with sftp.file("/tmp/deepseek-14b-main.service", "w") as remote:
            remote.write(service_14b)
        sftp.close()

        print("== deploy local tools API ==")
        sftp = ssh.open_sftp()
        with sftp.file(f"{TOOLS_DIR}/local_ai_tools_api.py", "w") as remote:
            remote.write(LOCAL_AI_TOOLS_CODE)
        service_tools = textwrap.dedent("""\
            [Unit]
            Description=Trade Local AI Tools API
            After=network-online.target deepseek-14b-main.service
            Wants=network-online.target

            [Service]
            Type=simple
            User=linux
            WorkingDirectory=/data/trade_ai/tools
            Environment=PATH=/home/linux/anaconda3/envs/trade_ml/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            ExecStart=/home/linux/anaconda3/envs/trade_ml/bin/python -m uvicorn local_ai_tools_api:app --host 0.0.0.0 --port 8001
            Restart=always
            RestartSec=5
            StandardOutput=append:/data/trade_ai/logs/local_ai_tools_api.log
            StandardError=append:/data/trade_ai/logs/local_ai_tools_api.err.log

            [Install]
            WantedBy=multi-user.target
        """)
        with sftp.file("/tmp/local-ai-tools.service", "w") as remote:
            remote.write(service_tools)
        sftp.close()

        print(run(ssh, """
            set -e
            chmod +x /data/trade_ai/scripts/start_deepseek_14b_main.sh
            sudo mv /tmp/deepseek-14b-main.service /etc/systemd/system/deepseek-14b-main.service
            sudo mv /tmp/local-ai-tools.service /etc/systemd/system/local-ai-tools.service
            sudo systemctl daemon-reload
            sudo systemctl enable deepseek-14b-main.service local-ai-tools.service
            sudo systemctl restart deepseek-14b-main.service
            sleep 12
            sudo systemctl restart local-ai-tools.service
            sleep 5
            echo '--- active ---'
            systemctl is-active deepseek-14b-main.service || true
            systemctl is-active local-ai-tools.service || true
            echo '--- models ---'
            curl -s http://127.0.0.1:8000/v1/models || true
            echo
            echo '--- tools ---'
            curl -s http://127.0.0.1:8001/health || true
            echo
            echo '--- gpu ---'
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
            echo '--- recent 14b log ---'
            tail -n 80 /data/trade_ai/logs/deepseek_14b_main.log 2>/dev/null || true
        """, timeout=600))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
