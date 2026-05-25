# AI PR Review Orchestrator — V2 and Beyond

Everything in this document was carved out of V1 to keep the initial loop
minimal. Each section is self-contained and can be picked up independently once
V1 is stable and idempotent.

---

## Multi-Phase Review Pipeline

V1 runs a single review phase with one reviewer. V2 adds config-driven phases
with multiple reviewers per phase.

```yaml
phases:
  - id: initial_360_review
    reviewers: [gemini_github, amazon_q_github]
    max_rounds: 1
    trigger_mode: parallel
    handle_mode: batch
    poll_interval_seconds: 30
    per_reviewer_timeout_seconds: 600
    phase_timeout_seconds: 900
    stop_when_empty: true
  - id: adversarial_loop
    reviewers: [claude_github, codex_github]
    max_rounds: 3
    trigger_mode: parallel
    handle_mode: batch
    stop_when_empty: true
```

State model changes:

- Add `phase_index` and `phase_id` to `RuntimeState`
- Add `active_reviewers: list[str]` and `reviewer_collected: dict[str, bool]`
- Add `archived_findings: dict[str, dict[str, int]]` for phase summaries
- State machine advances through phases sequentially

---

## Additional Reviewer Adapters

### Amazon Q GitHub

```yaml
amazon_q_github:
  enabled: true
  bot_logins:
    - "amazon-q-developer[bot]"
    - "aws-q-developer[bot]"
  trigger_comment: "/q review"
```

### Codex GitHub Reviewer

Use Codex as a reviewer (separate from its coder role).

### Claude GitHub Reviewer

Use Claude Code as a reviewer via `claude-code-action` or a trigger comment.

### Generic CLI Reviewer

Run an arbitrary CLI command that outputs findings in a normalized JSON format.
Useful for local linters, custom review scripts, or OSS review tools.

---

## Additional Coder Adapters

### Claude Code CLI

```python
class ClaudeCodeAdapter(CoderAdapter):
    # Uses `claude -p` with explicit file-output instruction
```

### Aider CLI

### OpenCode CLI

### Goose CLI

### Generic CLI

Configurable command that reads a task JSON from stdin or a file and writes
the result JSON to the output file.

---

## LLM Gateway Adapter

For simple chat-style coder or reviewer modes that don't need a dedicated CLI.

```yaml
llm_gateways:
  litellm:
    enabled: false
    base_url: null
  openrouter:
    enabled: false
    base_url: "https://openrouter.ai/api/v1"
```

A `generic_llm` adapter backed by LiteLLM or OpenRouter standardizes:

- Message format and model selection
- Timeout and retry behavior
- Structured JSON response extraction
- Provider auth configuration
- Token and secret redaction

The gateway adapter returns the same `AgentRunResult` and passes through the
same JSON validator. It is an adapter implementation detail, not a replacement
for the `CoderAdapter` / `ReviewerAdapter` interfaces.

---

## State Branch Storage

Migrate from PR-comment state to a dedicated orphan git branch when state
payloads grow beyond comment size limits (~65KB).

```
aipro-state/
  prs/123.json
  prs/124.json
```

Requirements:

- Create orphan branch if missing
- Fast-forward-only optimistic concurrency
- Retry once after non-fast-forward rejection
- PR comment becomes a pointer to the state branch commit
- State schema migration (version N -> N+1) on read

### State Compaction

When `handled_findings` exceeds 150 entries, archive completed phases into
summary counts:

```json
"archived_findings": {
  "initial_review": {"accepted": 3, "rejected": 2, "needs_human": 0}
}
```

---

## Finding Conflict Detection

When multiple reviewers operate on the same PR, their findings may contradict
each other.

### V2: Positional Conflict Detection

Group findings by `(path, line ± 5)`. If two findings in the same group come
from different reviewers:

- Assign both the same `conflict_group_id`
- Add a note to the coder prompt: "These findings may contradict each other.
  Evaluate independently."
- If the coder accepts both sides of a contradiction, reject the output and
  transition to `needs_human`

### V3: Semantic Conflict Detection

Use embeddings or LLM classification to detect semantically contradictory
suggestions even when they don't overlap positionally.

---

## Stagnation Detection

Track compact hashes in state to detect repeated failures:

```python
@dataclass
class PatchAttempt:
    head_sha: str
    diff_hash: str        # hash of actual file patch content
    finding_ids: list[str]
    result: Literal["tests_failed", "ci_failed", "review_rejected", "pushed"]
    failure_hash: str | None

@dataclass
class RebuttalAttempt:
    finding_id: str
    reply_hash: str
    reviewer: str
    result: Literal["posted", "reviewer_repeated", "needs_human"]
```

Rules:

- Same diff hash for same findings after prior failure -> `needs_human`
- Same rebuttal for a finding the reviewer repeats -> `needs_human`
- Same check failure with same compact failure summary after fix attempt ->
  `needs_human`
- Same `(head_sha, phase_id, round_index, finding_ids)` with no new findings
  and no new patch -> `done` or `needs_human`

Normalize diff hashes by excluding volatile metadata. Normalize failure hashes
from check name + failing command + stable excerpt.

---

## Prompt Size Management

When the total prompt (findings + diff + instructions) exceeds
`safety.max_prompt_tokens`:

1. Include higher-severity findings first
2. Prioritize files with the most review comments
3. Include scoped diffs for mentioned files only
4. Include dependency hints (imports, call sites, type references)
5. Split remaining findings into separate batches within the same round

The prompt must not imply the agent is forbidden from reading the rest of the
repo. It is a starting packet, not a context boundary.

---

## Automatic CI Fix

V1 treats CI failure as terminal (`needs_human`). V2 adds an optional CI-fix
flow:

1. On CI failure, collect failing check names and relevant log lines
2. Invoke the coder with a CI-fix task (different prompt template)
3. Apply the same validation, commit, and push cycle
4. Cap CI-fix attempts (e.g., max 2)
5. If the fix attempt fails or produces the same failure hash, `needs_human`

---

## Automatic History Cleanup

V1 never force-pushes or squashes. V2 adds an opt-in, maintainer-triggered
cleanup command:

```
aipro squash --pr 123
```

- Produces a proposed squash plan (preview only)
- Requires explicit confirmation
- Disabled by default
- Never runs on untrusted branches or fork PRs

---

## Token-Level Cost Tracking

V1 captures token counts from the coder's self-reported `token_usage`. V2 adds:

- Per-adapter token tracking (coder + reviewer + gateway)
- Dollar-cost estimation based on model pricing tables
- Configurable cost budgets per PR and per run
- Cost summary in the final PR comment

---

## State Schema Migration

When the state `version` field changes between releases:

```python
def migrate_state(raw: dict) -> RuntimeState:
    v = raw.get("version", 1)
    while v < CURRENT_VERSION:
        raw = MIGRATIONS[v](raw)
        v += 1
    return RuntimeState(**raw)
```

Each migration is a pure function from dict to dict. Unknown future fields are
preserved (forward compatibility).

---

## Multi-Repo Dashboard

A web UI or CLI aggregation command that shows orchestrator status across
multiple repositories:

- Active PRs and their current state
- Cost summaries
- Error rates
- Average time-to-done

---

## Platform Support

### GitLab / Bitbucket

Replace the GitHub client with platform-specific implementations. The state
machine and adapter interfaces are platform-agnostic by design.

### Webhook Server

Replace the GitHub Actions execution model with a standalone webhook server for
lower latency and finer-grained control. Requires durable job queue (Redis,
SQS, etc.) and persistent state (database replaces state branch/comment).

---

## Public Repository Assessment

Reference projects for implementation details:

| Project | Useful For |
|---------|-----------|
| [coding-review-agent-loop](https://github.com/wwind123/coding-review-agent-loop) | Review-loop behavior patterns |
| [codex-review-action](https://github.com/gersmann/codex-review-action) | GitHub Action packaging, deduplication |
| [claude-review-loop](https://github.com/hamelsmu/claude-review-loop) | Treating reviewer comments as hypotheses |
| [codex-action](https://github.com/openai/codex-action) | Codex GitHub Action integration |
| [claude-code-action](https://github.com/anthropics/claude-code-action) | Claude Code GitHub integration |
| [run-gemini-cli](https://github.com/google-github-actions/run-gemini-cli) | Gemini CLI from GitHub Actions |
| [pr-agent](https://github.com/The-PR-Agent/pr-agent) | PR diff plumbing, provider abstractions |
| [openreview](https://github.com/vercel-labs/openreview) | Sandboxed AI review infrastructure |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | Future OSS coder adapter |
| [Optio](https://optio.host/) | Full agentic platform (heavier alternative) |

Recommendation: build a small orchestrator with explicit adapters. Do not fork a
full platform as the core.
