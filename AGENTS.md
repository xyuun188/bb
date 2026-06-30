## Hindsight Project Memory

This repository has a project-specific Hindsight memory bank:

- Bank ID: `bb`
- Project root: `F:\bb`
- Scope: only memories about this project. Do not store this project's memories in `codex` or another project bank.
- Hindsight API/MCP base: `http://45.207.197.48:18888`
- Hindsight MCP endpoint: `http://45.207.197.48:18888/mcp/`
- Hindsight control panel: `http://45.207.197.48:19999/zh-CN`

Use the persistent CODEX++ MCP server `hindsight_bb` for this repository. The old `.codex-memory` scripts are not present in this checkout and must not be called.

At the beginning of a session, do not run broad live recall by default. Use `hindsight_bb.recall` only when current context is needed, with `budget="low"` and a small `max_tokens` limit first. Escalate to broader recall only when the user explicitly asks for deeper historical context.

Durable project changes are retained automatically by the user-level CODEX++ watcher and local Git hooks under `C:\Users\Administrator\.codex\hindsight-memory`. Do not duplicate this by saving every turn manually.

## Required Post-Change Workflow

After every completed code change in this repository, update all three places before finishing:

1. Verify the change with the smallest relevant tests/checks.
2. Sync the code to the online server, normally with:

```powershell
python scripts/sync_to_online_server.py --split-services
```

Use narrower options such as `--skip-restart`, `--include-tests`, or `--only` only when the task explicitly calls for staged validation.

3. Commit and push the change to GitHub from `F:\BB`:

```powershell
rtk git status --short
rtk git add <changed files>
rtk git commit -m "<concise change summary>"
rtk git push origin main
```

4. Update project memory for meaningful durable facts with the `bb` Hindsight bank. Prefer the configured CODEX++ automation/manual sync path, and never store secrets:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\.codex\hindsight-memory\sync-project-memory.ps1" -ProjectRoot "F:\BB" -Event "manual"
```

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# BAD:
git add . && git commit -m "msg" && git push

# GOOD:
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Codex sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to AGENTS.md
rtk init --global       # Add RTK to ~/.Codex/AGENTS.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->
## Codex Automatic Hindsight Workflow

For this repository, use the project-specific Hindsight MCP server `hindsight_bb` and bank `bb` automatically.
Hindsight reads and writes must use `http://45.207.197.48:18888/mcp/bb/` by default.

Do not run old `D:\code\Hindsight\codex_memory_*.ps1` scripts; that directory is not part of the current setup. Project changes are retained by user-level CODEX++ automation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\.codex\hindsight-memory\sync-project-memory.ps1" -ProjectRoot "F:\BB" -Event "manual"
```

Only call live recall when it is needed for the current task. Prefer `hindsight_bb.recall` with `budget="low"` and `max_tokens` around `1200` first; avoid `budget="high"` at session start because it commonly takes 30+ seconds on this remote Hindsight server.

Keep all memories for this repository in `bb`; do not store this project's facts in the global `codex`, `manju`, or `MetaCode` banks.
<!-- BEGIN CODEX SEMBLE RTK -->
## Project Tooling

- Semble is available globally through the Codex MCP server. Use Semble first for semantic code discovery in this project.
- Use mcp__semble.search with repo="F:\\bb" for broad codebase questions.
- If the current Codex session has not hot-loaded `mcp__semble`, use the local CLI fallback before broad `rg`/file reads:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:SEMBLE_CACHE_LOCATION='F:\SemblevsRTK\cache\semble'
$env:SEMBLE_MODEL_NAME='F:\SemblevsRTK\models\potion-code-16M'
& 'F:\SemblevsRTK\.venvs\semble\Scripts\semble.exe' search '<query>' 'F:\bb' --top-k 5
```

- For any request asking "where", "how", architecture flow, similar code, risk/strategy/service relationships, or unknown symbols, run Semble search first and use returned file locations to narrow follow-up reads.
- Codebase Memory MCP is available globally as `codebase_memory`; use repo path `F:\BB` for indexing, architecture queries, call-path tracing, code snippets, graph searches, and change detection.
- `Codex-CodebaseMemory-AutoIndex` refreshes this project's Codebase Memory graph every 30 minutes and at logon. Already-open Codex sessions cannot hot-load native MCP tools; use the fallback script `C:\Users\Administrator\.codex\codebase-memory\invoke-codebase-memory-tool.ps1` if needed.
- Treat Codebase Memory as structural code intelligence. Keep durable project decisions and preferences in `hindsight_bb`, not in another project's memory bank.
- RTK is available globally from F:\SemblevsRTK\bin\rtk.exe. Prefer RTK for supported shell commands that produce large output.
- Do not force RTK onto PowerShell-only commands such as Get-ChildItem, Get-Content, Select-String, or Test-Path.
<!-- END CODEX SEMBLE RTK -->
