#!/usr/bin/env python3
"""Minimal OpenAI-compatible Qwen3.8-27B Transformers inference service.

The target host has one A100 and a single-request budget.  This service loads
the official Qwen3.8-27B weights with bitsandbytes NF4 and deliberately keeps
the API small: callers get the same model/list/chat contract used by the
existing decision pipeline without a second model server or legacy aliases.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)
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
        self.model_id = args.model_id
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
        self.model = AutoModelForMultimodalLM.from_pretrained(
            args.model,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map="auto",
            max_memory={0: "34GiB", "cpu": "64GiB"},
            low_cpu_mem_usage=True,
        )
        if self.model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
            raise RuntimeError("loaded model class is not the official Qwen3.8 architecture")
        if args.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                args.adapter_path,
                is_trainable=False,
            )
        self.model.eval()
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            self.input_device = next(self.model.parameters()).device
        self.lock = threading.Lock()

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
        generation_kwargs = {
            **encoded,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "temperature": max(request.temperature, 1e-5),
            "top_p": request.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        generated = output[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="BB Qwen3.8-27B target", docs_url=None, redoc_url=None)
    started_at = time.time()

    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        return {"status": "ok", "model_id": runtime.model_id, "uptime_seconds": time.time() - started_at}

    @app.get("/health/ready")
    def health_ready() -> dict[str, Any]:
        return {"status": "ready", "model_id": runtime.model_id}

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
        if not runtime.lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="single model concurrency limit reached")
        try:
            content = runtime.generate(request)
        finally:
            runtime.lock.release()
        now = int(time.time())
        prompt_tokens = 0
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
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(content.split()),
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
    args = parser.parse_args()
    if args.max_concurrency != 1:
        parser.error("target Transformers service only permits max-concurrency=1")
    return args


def main() -> None:
    args = parse_args()
    runtime = Runtime(args)
    uvicorn.run(build_app(runtime), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
