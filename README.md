# BB Quantitative Trading Platform

BB is an OKX demo/paper quantitative trading platform. The system is kept
paper-only until the authoritative fee-after-return gates pass; no model
endpoint or configuration can enable live routing by itself.

## Runtime Topology

```text
market facts -> analysis scheduler -> Qwen3.8-27B carrier -> risk gates
                                                      -> paper executor
OKX authoritative fills/fees/funding -> settlement -> training evidence
```

- One local LLM service: `qwen3.8-27b` on model-host port `8000`.
- Platform tunnel: `127.0.0.1:18000 -> model-host:8000`.
- Quant API: `127.0.0.1:18001 -> model-host:8101`.
- The five expert roles are prompt-scoped views of the same carrier, not five
  independent model votes.
- High-risk entry review is an independent public HTTPS cloud route. Missing,
  local, stale, or incomplete reviewer identity blocks the entry.
- The old 14B and local DeepSeek services are retired and must remain stopped.

## Promotion Rules

Training and model promotion optimize realized net return after fees, funding,
slippage, and execution costs. Win rate is not a promotion criterion. A model
must provide immutable dataset lineage, walk-forward evidence, regime and
symbol holdouts, authoritative OKX settlement evidence, and a positive risk-
adjusted return distribution before it can leave paper mode.

The target model cannot be deployed from a model name alone. The model host
must first produce a verified candidate manifest containing the immutable
repository revision, config/tokenizer/weight hashes, runtime versions, and at
least 20 successful inference probes with zero OOM, timeout, or JSON-contract
failures.

## Local Checks

Run from the repository root with `uv`:

```powershell
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q ai_brain config core data_feed db executor models risk_manager scripts services web_dashboard
uv run python scripts/security_secret_scan.py --fail-on high .
uv run python scripts/audit_profit_integrity_architecture.py
```

All checks must pass before deployment. Use the read-only model-server audit
before any migration:

```powershell
uv run python scripts/run_phase3_model_server_readiness_audit.py
```

The audit is intentionally fail-closed. It will not download weights, stop an
old service, create a manifest, or enable trading.

## Deployment Order

1. Download and hash the approved Qwen3.8-27B artifact on the model host.
2. Run the isolated runtime probe and build `target_model_candidate.json`.
3. Validate the manifest and install the single target service transactionally.
4. Stop and disable all retired 14B/DeepSeek services.
5. Sync the platform source and loopback tunnels; verify dashboard, market
   analysis, settlement, and training heartbeats.
6. Observe paper mode for at least 24-72 hours. Live routing remains disabled
   until every return and risk gate is green.

## Security

Credentials belong in the ignored account-info files or deployment environment,
never in source, tests, documentation, logs, or committed manifests. The
dashboard and model tunnels are separate services; an HTTP health response is
not evidence that trading is authorized.
