## Hindsight Project Memory

This repository has a project-specific Hindsight memory bank:

- Bank ID: `bb`
- Project root: `F:\bb`
- Scope: only memories about this project. Do not store this project's memories in `codex` or another project bank.

At the beginning of a Codex session for this repository, recall project memory:

```powershell
powershell.exe -ExecutionPolicy Bypass -File ".\.codex-memory\recall_project_memory.ps1" -Query "What should I remember about this project?"
```

Prefer reading `PROJECT_MEMORY.md` first when it exists. If it is missing or stale, refresh it:

```powershell
powershell.exe -ExecutionPolicy Bypass -File ".\.codex-memory\refresh_project_memory.ps1"
```

When the user asks to remember something about this project, add it here:

```powershell
powershell.exe -ExecutionPolicy Bypass -File ".\.codex-memory\add_project_memory.ps1" -Content "memory text"
```

After adding durable project memory, refresh `PROJECT_MEMORY.md` with `.codex-memory\refresh_project_memory.ps1`.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# 鉂?Wrong
git add . && git commit -m "msg" && git push

# 鉁?Correct
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
When the user asks to remember something about this project, add it here:

```powershell
powershell.exe -ExecutionPolicy Bypass -File ".\.codex-memory\add_project_memory.ps1" -Content "memory text"
```
## Codex Automatic Hindsight Workflow

For this repository, use the project-specific Hindsight bank `bb` automatically.

At the beginning of every Codex session for this repository, recall memory before substantive work. The default command is the fast path: it reads `PROJECT_MEMORY.md` and skips server-side live recall to avoid shell timeouts.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\code\Hindsight\codex_memory_start.ps1" -ProjectRoot "F:\bb"
```

Use `-LiveRecall` only when a fresh Hindsight server recall is explicitly needed. Use `-RefreshSnapshot` only when rebuilding `PROJECT_MEMORY.md` is explicitly needed.

During work, save durable project facts to Hindsight when they would prevent repeated work later: architecture decisions, important paths, service ports, tested fixes, setup steps, user preferences for this project, and non-obvious debugging results. Do not save secrets, one-off chatter, temporary failed guesses, or bulky command output.

Before finishing a meaningful task in this repository, summarize and distill the useful long-term facts before saving them to the `bb` bank. Do not store the whole chat verbatim. The default save path is fast: it saves the provided distilled notes asynchronously and does not refresh the large snapshot.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\code\Hindsight\codex_memory_summarize_and_save.ps1" -ProjectRoot "F:\bb" -Content "session notes to distill"
```

Use `-UseReflect` only when Hindsight should do an extra cloud-model distillation pass before saving. Use `-RefreshSnapshot` only when the saved memory must immediately appear in `PROJECT_MEMORY.md`.

Shortcut:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\code\Hindsight\codex_memory_end.ps1" -ProjectRoot "F:\bb" -Content "session notes to distill"
```

Keep all memories for this repository in `bb`; do not store this project's facts in the global `codex` bank.
<!-- BEGIN CODEX SEMBLE RTK -->
## Project Tooling

- Semble is available globally through the Codex MCP server. Use Semble first for semantic code discovery in this project.
- Use mcp__semble.search with repo="F:\\bb" for broad codebase questions.
- RTK is available globally from F:\SemblevsRTK\bin\rtk.exe. Prefer RTK for supported shell commands that produce large output.
- Do not force RTK onto PowerShell-only commands such as Get-ChildItem, Get-Content, Select-String, or Test-Path.
<!-- END CODEX SEMBLE RTK -->
