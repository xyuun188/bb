"""Shared endpoint constants for tests that exercise the approved dual-14B routing."""

PUBLIC_MODEL_TEST_HOST = "10.0.0.5"
QWEN_PUBLIC_TEST_BASE = f"http://{PUBLIC_MODEL_TEST_HOST}:21840/v1"
DEEPSEEK_PUBLIC_TEST_BASE = f"http://{PUBLIC_MODEL_TEST_HOST}:21842/v1"

LOCAL_QWEN_TEST_BASE = "http://127.0.0.1:8000/v1"
LOCAL_DEEPSEEK_TEST_BASE = "http://127.0.0.1:8002/v1"
