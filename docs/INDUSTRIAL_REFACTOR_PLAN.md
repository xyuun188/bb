# Industrial Refactor Plan

This project is a trading system, so the refactor must reduce operational risk
before changing strategy behavior. Work is split into small, verifiable phases.

## Active Engineering Standards

- Clean architecture: keep API, orchestration, policy, execution, persistence,
  and external integrations behind explicit boundaries.
- Secure configuration: secrets live in environment variables or ignored local
  files only; source code and logs must never expose keys, tokens, passphrases,
  passwords, or authorization headers.
- Maintainability: prefer small policy modules, typed dataclasses, focused
  services, and tests around trading invariants.
- Formatting: use Black-compatible formatting and Ruff lint rules.
- Type safety: add type hints to new and touched Python code; expand mypy scope
  gradually as legacy modules are split.
- Observability: use structured logs with redaction; operational errors should
  include context without leaking secrets.
- Async reliability: wrap external calls with timeouts, bounded concurrency, and
  explicit fallback behavior.
- UI quality: keep dashboard states dense, readable, responsive, and consistent
  with the existing dark operational style.
- Model operations: keep remote Qwen/local tools and online review clearly
  separated, health-checked, timeout-bounded, and non-thinking where configured.

## Phase 1: Safety and Tooling Baseline

- Status: implemented for the touched code path.
- Added project-level Black, Ruff, Mypy, and Pytest configuration.
- Added a source secret scanner that skips local secret containers but fails on
  hardcoded or logged secrets.
- Removed known secret-prefix logging from live startup.
- Introduced shared redaction utilities for logs and diagnostics.
- Hardened settings export and `.env` writes: recursive redaction, masked-value
  write prevention, key validation, newline rejection, and atomic replacement.
- Default dashboard host is now localhost; explicit environment override is
  still available when remote access is intentionally required.

## Phase 2: Architecture Boundaries

- Status: in progress.
- Continue extracting entry evidence, sizing, exit policy, execution, and sync
  behavior out of the legacy trading orchestrator.
- Keep TradingService as orchestration only: no direct exchange logic, no direct
  UI shaping, and no raw model-service deployment behavior.
- Add focused tests for each extracted policy module.
- Shared entry-signal extraction now recovers ML/server-profit/timeseries
  evidence from `opportunity_score` and `evidence_score` so strategy replay does
  not lose data when older raw tool blocks are missing.

## Phase 3: Model and Server Resilience

- Status: in progress.
- Centralize local/remote model endpoint configuration, token limits, timeouts,
  non-thinking controls, and fallback policy.
- Add health-check scripts that verify Qwen3 vLLM, local-ai-tools, and online
  review separately without printing secrets.
- Record model latency, timeout, parse-failure, and fallback counters.
- Qwen3/DeepSeek-R1 thinking controls are centralized in `core.model_runtime`.
- Expert, decision-maker, and batch-expert completion token limits are hard
  capped centrally at 360/520/700 to prevent slow local 32B responses.
- Remote server credentials are loaded through a redacted parser instead of
  duplicated script-local parsing.

## Phase 4: Dashboard and UX Quality

- Status: in progress.
- Consolidate dashboard API payloads into typed response builders.
- Keep trading controls explicit and guarded for live/demo/paper modes.
- Improve table density, wrapping, dark-mode contrast, and responsive layout.
- Strategy replay record evidence now renders as a fixed two-line compact rail
  with stable widths, so the AI/ML/shadow column does not grow row spacing.

## Phase 5: Verification Gate

- Run secret scan, py_compile, focused pytest, and available lint/format checks.
- Start local services only when needed for UI/API verification.
- Save durable project decisions to the `bb` Hindsight bank.
