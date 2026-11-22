# Intelli-Speech Agent Workflow

## Roles

- VS Code Copilot is the Lead Architect and Orchestrator.
- VS CODE Copilot will handover the execution (building part completely to opencode agents after hitting the credit limits 55% and focus on monitoring the codebase and guiding opencode.)
- Before handovering the execution completely to the opencode agent, VS CODE Copilot has to tell which opencode session has the full context about the project so that the user can use to continue chat and building application
- OpenCode CLI is the parallel implementation and investigation worker pool.
- OpenCode may use only providers and models that are explicitly listed as available and free in `opencode.json`.
- OpenCode must not connect to the user's GitHub Copilot account or credentials.

## Existing Agent Configuration

The repository's `opencode.json` defines:

- `plan`: read-only architecture analysis, technical specifications, edge-case discovery, and runtime compatibility research.
- `build`: approved implementation, focused refactors, boilerplate, test integration, and documentation updates.
- `repository-explorer`: read-only repository search and implementation mapping.
- `runtime-researcher`: read-only compatibility and runtime investigation.
- `test-author`: focused test creation within approved test paths.
- `documentation-editor`: directly related Markdown documentation updates.
- `code-reviewer`: read-only review of an assigned diff or file set.

Only provider/model combinations verified by `opencode models` may be assigned in `opencode.json`. Before changing a model, run:

```powershell
opencode models
```

## Lead Architect Responsibilities

VS Code Copilot retains responsibility for:

- Approving scope and deciding whether work should remain direct or be delegated.
- Preserving existing notebooks and original datasets.
- Preventing overlapping edits and resolving Git conflicts.
- Reviewing final diffs and validation results.
- Committing and pushing only when the user explicitly requests it or has established the repository workflow.

OpenCode may make changes only inside work assigned to it.

## Work Routing

### Handle directly

Handle a task directly when it is small, cross-cutting, requires live Colab/Drive state, involves Git or GitHub synchronization, or needs decisions from incomplete runtime evidence.

Examples:

- Running or editing Google Colab setup.
- Connecting VS Code to a live Colab runtime.
- Resolving Git conflicts or reviewing large notebook diffs.
- Making architectural decisions that span several modules.
- Handling secrets, credentials, or `.env` files.

### Delegate to `plan`

Delegate read-only analysis when independent investigation can improve correctness before implementation.

Examples:

- Mapping existing dataset-loading, enhancement, baseline, and evaluation code.
- Investigating dependency or runtime compatibility.
- Producing an implementation specification.
- Finding edge cases or design risks.

`plan` must not edit source files or notebooks and must distinguish:

- implemented behavior;
- verified behavior;
- proposed behavior;
- unverified assumptions.

### Delegate to `build`

Delegate implementation only after the target files and acceptance criteria are explicit.

Examples:

- A small Python module with focused tests.
- A narrowly scoped frontend implementation.
- Boilerplate that follows an approved existing pattern.
- A reviewed notebook helper script that does not replace existing experiments.

`build` must stop if runtime state or requirements are unverified.

### Delegate specialists

Use specialists only for disjoint, bounded work:

- `repository-explorer` for read-only codebase searches.
- `runtime-researcher` for read-only package, Python, Colab, and FRCRN compatibility analysis.
- `test-author` for focused tests in existing test locations.
- `documentation-editor` for directly related Markdown.
- `code-reviewer` for read-only review of an assigned diff.

Do not delegate a continuous trace across multiple interconnected files to several agents. Follow the trace directly or assign one lead agent.

### Model assignment (free tier only)

`opencode.json` assigns per role (verified with `opencode models`, 2026-11-22): `plan`/`build` → `longcat-2.5-free`; `test-author`/`runtime-researcher` → `flint-40b-free`; `repository-explorer`/`documentation-editor` → `nemotron-3.5-lightning-free`; `code-reviewer` → `tidepool-alpha-free`. If a model hits its free limit, switch the role to another free model listed by `opencode models`. No paid providers; OpenRouter is not configured.

### Codex CLI (second-opinion worker)

Codex (ChatGPT sign-in, limited free quota) is used sparingly for high-value read-only work: reviewing statistics/feature code, independent re-checks, and drafting tests.

- Default: `codex exec --sandbox read-only "<scoped task>"`.
- Edits only in a separate Git worktree; the lead agent reviews and merges.
- Same data rules as OpenCode: no transcripts, speaker IDs, audio, HF tokens, or `.env` contents in prompts — code and synthetic fixtures only.
- Codex never commits or pushes.

## Parallel Sessions

For independent subtasks:

1. Define disjoint file scope and a clear acceptance criterion.
2. Use a separate session title for every independent task.
3. Use separate Git worktrees for tasks that need simultaneous edits to the same branch.
4. Do not attach or pass `.env` files, credentials, tokens, private keys, or secrets.
5. Do not expose full audio datasets or model checkpoints to Git or unrelated tasks.
6. Never run agents in parallel on the same notebook or shared source file.
7. Never have two agents commit or push concurrently.

Example commands must be run by the Lead Architect with explicit scope:

```powershell
opencode run --agent plan --title "Audit FRCRN runtime" "Inspect the repository and report the smallest verified Colab runtime restoration plan."

opencode run --agent build --title "Add audio I/O helper" --file scripts --session <session-id> "Implement the approved helper in disjoint files only."
```

The `--dir` and `--worktree` flags are not part of every installed OpenCode CLI version. If unavailable, use a pre-created Git worktree and run OpenCode from that worktree directory.

Use `--session <id>` or `--continue` only for sequential follow-up work on the same bounded task.

## Protected Repository Areas

Protect these areas unless the user explicitly approves a change:

- Original datasets and Google Drive data.
- Paired dataset v3 files and metadata.
- Existing notebooks and their historical experiment outputs.
- Audio model checkpoints and generated experiment artifacts.
- `.env` files and credential configuration.
- The existing frontend's clearly-labelled demo behavior.

Notebook edits must append or update only the explicitly assigned cell. Do not replace, reorder, merge, or delete existing cells unless that is the approved task.

## Required Workflow

Before editing:

1. Inspect relevant source, documentation, and current Git status.
2. State the affected files and acceptance criterion.
3. Identify dependencies or runtime assumptions.
4. Start from the smallest verifiable change.

Before wrapping up:

1. Inspect `git diff` for every changed file.
2. Run the smallest relevant test or validation command.
3. Validate notebook JSON and structure after notebook edits.
4. Run a frontend build when frontend TypeScript or Vite sources change.
5. Confirm the original datasets and historical notebook experiments remain untouched.
6. Report what was implemented, what was validated, and what remains unverified.

Do not claim success without actual command or runtime output.

## Git Safety

- Never force-push.
- Never reset or revert unrelated work.
- Never commit secrets or large datasets/checkpoints.
- Never overwrite a notebook in an active Colab session.
- Commit and push only when explicitly requested or by an established workflow that the user has approved.
- Do not include temporary agent sessions or files in repository commits.

## Security and Data Handling

- Never pass `.env`, API keys, OAuth tokens, GitHub credentials, or private URLs to OpenCode.
- Do not grant OpenCode provider access outside `opencode`.
- Keep Google Drive dataset paths and transcripts out of prompts unless required and already project-contextual.
- Do not expose sample-level transcripts or sensitive metadata unnecessarily.
