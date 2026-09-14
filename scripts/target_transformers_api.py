#!/usr/bin/env python3
"""Minimal OpenAI-compatible Qwen3.8-27B Transformers inference service.

The target host has one A100 and a single-request budget.  This service loads
the official Qwen3.8-27B weights with bitsandbytes NF4 and deliberately keeps
the API small: callers get the same model/list/chat contract used by the
existing decision pipeline without a second model server or legacy aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("bb.target_transformers_api")


class GenerationTimeoutError(TimeoutError):
    """A bounded request expired while generation continued in the worker."""


class GenerationBusyError(RuntimeError):
    """The previous timed-out generation is still draining."""


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # The target carrier is reserved for the compact trading-diagnostic
    # contract. Larger legacy completions create long single-worker drains.
    max_tokens: int = Field(default=96, ge=1, le=256)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


def _text_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


class Runtime:
    def __init__(self, args: argparse.Namespace):
        # Keep matmul on Tensor Cores and avoid the slow FP32 fallback that is
        # otherwise easy to hit on an A100 when the process is launched by
        # systemd without the interactive shell environment.
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model_id = args.model_id
        self.adapter_path = str(args.adapter_path or "").strip()
        if args.require_adapter and not self.adapter_path:
            raise RuntimeError("Qwen3.8-27B target service requires a verified FinQuant adapter")
        self.context_length = args.context_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=False,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs = {
            "trust_remote_code": False,
            "dtype": torch.bfloat16,
            "quantization_config": quantization,
            "device_map": "auto",
            "max_memory": {0: "34GiB", "cpu": "64GiB"},
            "low_cpu_mem_usage": True,
        }
        configured_attention = os.environ.get("BB_TARGET_ATTN_IMPLEMENTATION")
        if configured_attention is None:
            # FlashAttention-2 is materially faster on the A100 when the
            # installed wheel is available.  Keep SDPA as the deterministic
            # fallback for a clean target-inference environment.
            configured_attention = (
                "flash_attention_2"
                if importlib.util.find_spec("flash_attn")
                else "sdpa"
            )
        requested_attention = str(configured_attention).strip()
        if requested_attention:
            model_kwargs["attn_implementation"] = requested_attention
        try:
            self.model = AutoModelForMultimodalLM.from_pretrained(
                args.model,
                **model_kwargs,
            )
            self.attention_implementation = requested_attention or "auto"
        except (TypeError, ValueError) as exc:
            # Some Transformers/Qwen builds do not expose the generic
            # attention selector. Retry without it, but never hide an OOM or
            # another runtime failure that would make the service unsafe.
            if "attn_implementation" not in model_kwargs:
                raise
            logger.warning(
                "requested attention implementation unavailable; using model default",
                extra={"requested": requested_attention, "error": str(exc)[:240]},
            )
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                args.model,
                **model_kwargs,
            )
            self.attention_implementation = "auto"
        if self.model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
            raise RuntimeError("loaded model class is not the official Qwen3.8 architecture")
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False,
            )
        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = True
        self.fast_path = self._detect_fast_path()
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            self.input_device = next(self.model.parameters()).device
        self.lock = threading.Lock()
        # The A100 target intentionally runs one generation at a time. Keep
        # the queue short and make limits tunable without rebuilding the image.
        # A timed-out generation is isolated from the HTTP request; the process
        # is not killed and restarted for one slow prompt.
        self.max_queue_wait_seconds = _env_float(
            "BB_TARGET_QUEUE_WAIT_SECONDS", 2.0, minimum=0.5, maximum=30.0
        )
        self.generation_timeout_seconds = _env_float(
            "BB_TARGET_GENERATION_TIMEOUT_SECONDS", 12.0, minimum=8.0, maximum=120.0
        )
        # Keep the runtime contract identical to the generation boundary.
        # Older unit files may still export 128/256; accepting those values
        # makes readiness report a limit the carrier will never honor and
        # encourages callers to enqueue unnecessarily long generations.
        self.max_new_tokens = _env_int(
            "BB_TARGET_MAX_NEW_TOKENS", 32, minimum=8, maximum=96
        )
        self.warmup_timeout_seconds = 1800.0
        self.warmup_complete = False
        self.warmup_error: str | None = None
        self.warmup_started_at: float | None = None
        self.warmup_finished_at: float | None = None
        self._generation_state_lock = threading.Lock()
        self._active_generation_future: Future[str] | None = None
        self._generation_timeouts = 0
        self._last_generation_seconds: float | None = None
        self._last_prompt_tokens = 0
        self._last_generation_tokens = 0
        self._response_cache_ttl_seconds = _env_float(
            "BB_TARGET_RESPONSE_CACHE_SECONDS", 2.0, minimum=0.0, maximum=15.0
        )
        self._response_cache: dict[str, tuple[float, str]] = {}
        self._response_cache_lock = threading.Lock()
        self._response_cache_hits = 0
        self._generation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen-generation",
        )

    def generation_busy(self) -> bool:
        with self._generation_state_lock:
            future = self._active_generation_future
            return bool(future is not None and not future.done())

    def _clear_generation_future(self, future: Future[str]) -> None:
        with self._generation_state_lock:
            if self._active_generation_future is future:
                self._active_generation_future = None

    @staticmethod
    def cache_key(request: ChatRequest) -> str:
        payload = {
            "model": request.model,
            "messages": [
                {"role": item.role, "content": item.content}
                for item in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "chat_template_kwargs": request.chat_template_kwargs,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_cached_response(self, key: str) -> str | None:
        if self._response_cache_ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._response_cache_lock:
            item = self._response_cache.get(key)
            if item is None:
                return None
            created_at, content = item
            if now - created_at > self._response_cache_ttl_seconds:
                self._response_cache.pop(key, None)
                return None
            self._response_cache_hits += 1
            return content

    def cache_response(self, key: str, content: str) -> None:
        if self._response_cache_ttl_seconds <= 0:
            return
        now = time.monotonic()
        with self._response_cache_lock:
            self._response_cache[key] = (now, content)
            if len(self._response_cache) > 32:
                oldest = min(self._response_cache, key=lambda item: self._response_cache[item][0])
                self._response_cache.pop(oldest, None)

    def _generate_with_timeout(self, request: ChatRequest, timeout: float) -> str:
        if self.generation_busy():
            raise GenerationBusyError("previous generation is still draining")
        started = time.perf_counter()
        future: Future[str] = self._generation_executor.submit(self.generate, request)
        with self._generation_state_lock:
            self._active_generation_future = future
        future.add_done_callback(self._clear_generation_future)
        try:
            result = future.result(timeout=max(float(timeout), 1.0))
            self._last_generation_seconds = round(time.perf_counter() - started, 3)
            return result
        except FutureTimeout:
            self._generation_timeouts += 1
            elapsed = round(time.perf_counter() - started, 3)
            self._last_generation_seconds = elapsed
            logger.error(
                "bounded model generation timed out; keeping process alive",
                extra={
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": round(float(timeout), 3),
                    "max_new_tokens": getattr(request, "max_tokens", None),
                },
            )
            # A running Transformers generation cannot be safely interrupted in
            # a Python thread. Leave it draining in the single worker and make
            # subsequent requests fail fast until it completes.
            raise GenerationTimeoutError(
                f"model generation exceeded {float(timeout):.1f}s"
            ) from None

    def warmup(self) -> None:
        """Compile the representative path before accepting traffic."""

        self.warmup_started_at = time.time()
        request = ChatRequest(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": '/no_think\nReturn exactly this JSON: {"status":"ok"}',
                }
            ],
            temperature=0.0,
            max_tokens=8,
            chat_template_kwargs={"enable_thinking": False},
        )
        try:
            with self.lock:
                self._generate_with_timeout(request, self.warmup_timeout_seconds)
        except Exception as exc:  # pragma: no cover - host-runtime diagnostic.
            self.warmup_error = f"{type(exc).__name__}: {exc}"[:240]
        else:
            self.warmup_complete = True
        finally:
            self.warmup_finished_at = time.time()

    @staticmethod
    def _detect_fast_path() -> dict[str, Any]:
        """Expose kernel availability instead of reporting green on a slow fallback."""

        modules = {
            "flash_linear_attention": bool(importlib.util.find_spec("fla")),
            "causal_conv1d": bool(importlib.util.find_spec("causal_conv1d")),
        }
        causal_error = ""
        if modules["causal_conv1d"]:
            try:
                import causal_conv1d  # noqa: F401
            except Exception as exc:  # pragma: no cover - host-runtime diagnostic.
                modules["causal_conv1d"] = False
                causal_error = f"{type(exc).__name__}: {exc}"[:240]
        return {
            "enabled": all(modules.values()),
            "modules": modules,
            "causal_conv1d_import_error": causal_error or None,
            "torch_cuda": bool(torch.cuda.is_available()),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda or ""),
        }

    def generate(self, request: ChatRequest) -> str:
        messages = [
            {"role": item.role, "content": _text_content(item.content)}
            for item in request.messages
        ]
        enable_thinking = request.chat_template_kwargs.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise HTTPException(
                status_code=400,
                detail="chat_template_kwargs.enable_thinking must be boolean",
            )
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        if not hasattr(encoded, "items"):
            encoded = {"input_ids": encoded}
        encoded = {
            key: value.to(self.input_device)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.shape[-1] > self.context_length:
            raise HTTPException(status_code=400, detail="prompt exceeds context length")
        # Enforce the carrier budget at the last boundary. This protects the
        # GPU when an older caller still sends max_tokens=320/1024.
        max_new_tokens = min(
            max(int(request.max_tokens), 1),
            int(getattr(self, "max_new_tokens", 96) or 96),
            96,
        )
        self._last_prompt_tokens = int(input_ids.shape[-1])
        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": request.temperature > 0,
            "top_p": request.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if request.temperature > 0:
            generation_kwargs["temperature"] = request.temperature
        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        generated = output[0, input_ids.shape[-1] :]
        # Real Transformers returns a tensor here; small test doubles may
        # return an already-decoded value. Keep diagnostics best-effort for
        # the latter without changing the production path.
        generated_shape = getattr(generated, "shape", None)
        self._last_generation_tokens = (
            int(generated_shape[-1]) if generated_shape is not None else 0
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="BB Qwen3.8-27B target", docs_url=None, redoc_url=None)
    started_at = time.time()

    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        return {"status": "ok", "model_id": runtime.model_id, "uptime_seconds": time.time() - started_at}

    @app.get("/health/ready")
    def health_ready() -> JSONResponse:
        status = (
            "warming_up"
            if not runtime.warmup_complete and not runtime.warmup_error
            else "ready"
            if runtime.warmup_complete and runtime.fast_path["enabled"]
            else "degraded"
        )
        payload = {
            "status": status,
            "model_id": runtime.model_id,
            "adapter_loaded": bool(runtime.adapter_path),
            "adapter_path": runtime.adapter_path or None,
            "inference_path": "fused" if runtime.fast_path["enabled"] else "torch_fallback",
            "performance": runtime.fast_path,
            "attention_implementation": runtime.attention_implementation,
            "warmup": {
                "complete": runtime.warmup_complete,
                "error": runtime.warmup_error,
                "started_at": runtime.warmup_started_at,
                "finished_at": runtime.warmup_finished_at,
            },
            "generation": {
                "busy": runtime.generation_busy(),
                "queue_wait_seconds": runtime.max_queue_wait_seconds,
                "timeout_seconds": runtime.generation_timeout_seconds,
                "max_new_tokens": runtime.max_new_tokens,
                "timeouts": runtime._generation_timeouts,
                "last_seconds": runtime._last_generation_seconds,
                "last_prompt_tokens": runtime._last_prompt_tokens,
                "last_generation_tokens": runtime._last_generation_tokens,
                "cache_ttl_seconds": runtime._response_cache_ttl_seconds,
                "cache_hits": runtime._response_cache_hits,
            },
        }
        return JSONResponse(status_code=200 if status == "ready" else 503, content=payload)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": runtime.model_id, "object": "model", "owned_by": "bb"}],
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest, raw_request: Request) -> JSONResponse:
        del raw_request
        if request.model != runtime.model_id:
            raise HTTPException(status_code=404, detail="unknown model")
        if not runtime.warmup_complete:
            raise HTTPException(
                status_code=503,
                detail="target model is warming up",
                headers={"Retry-After": "5"},
            )
        cache_key = runtime.cache_key(request)
        cached = runtime.get_cached_response(cache_key)
        if cached is not None:
            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": runtime.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": cached},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": runtime._last_prompt_tokens,
                        "completion_tokens": runtime._last_generation_tokens,
                        "total_tokens": runtime._last_prompt_tokens
                        + runtime._last_generation_tokens,
                    },
                    "bb_cache": "hit",
                }
            )
        if runtime.generation_busy():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "previous generation is still draining",
                    "retry_after_seconds": 2,
                },
                headers={"Retry-After": "2"},
            )
        if not runtime.lock.acquire(timeout=runtime.max_queue_wait_seconds):
            raise HTTPException(
                status_code=503,
                detail=(
                    "single model queue wait exceeded "
                    f"{runtime.max_queue_wait_seconds:.1f} seconds"
                ),
                headers={"Retry-After": "2"},
            )
        try:
            try:
                content = runtime._generate_with_timeout(
                    request,
                    runtime.generation_timeout_seconds,
                )
            except GenerationTimeoutError as exc:
                return JSONResponse(
                    status_code=504,
                    content={
                        "error": "model_generation_timeout",
                        "detail": str(exc),
                        "retry_after_seconds": 2,
                    },
                    headers={"Retry-After": "2"},
                )
            except GenerationBusyError:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "previous generation is still draining",
                        "retry_after_seconds": 2,
                    },
                    headers={"Retry-After": "2"},
                )
        finally:
            runtime.lock.release()
        runtime.cache_response(cache_key, content)
        now = int(time.time())
        prompt_tokens = runtime._last_prompt_tokens
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": now,
                "model": runtime.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": runtime._last_generation_tokens,
                    "total_tokens": prompt_tokens + runtime._last_generation_tokens,
                },
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--require-adapter", action="store_true")
    args = parser.parse_args()
    if args.max_concurrency != 1:
        parser.error("target Transformers service only permits max-concurrency=1")
    return args


def main() -> None:
    args = parse_args()
    runtime = Runtime(args)
    threading.Thread(
        target=runtime.warmup,
        name="qwen-startup-warmup",
        daemon=True,
    ).start()
    uvicorn.run(build_app(runtime), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
