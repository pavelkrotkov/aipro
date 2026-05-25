# AI PR Review Orchestrator — V1 Implementation Plan

## 1. Objective

Build a standalone Python CLI that coordinates one AI coding agent and one AI
reviewer around a GitHub pull request. The orchestrator treats reviewer feedback
as hypotheses, asks the coder to evaluate and fix valid findings, and manages
the full lifecycle through a deterministic state machine.

V1 proves the loop with the fewest moving parts:

- One primary coding agent: `codex_cli`
- One reviewer adapter: `gemini_github`
- One fix iteration per workflow run
- One coherent fix commit per iteration
- Trusted same-repository PR branches only
- Bot-authored review threads only for auto-reply and auto-resolution

---

## 2. Core Design Principle

Separate orchestration from reasoning.

The LLM/coding agent:

- Inspects code
- Evaluates comments
- Makes patches
- Writes explanations
- Runs targeted tests
- Returns structured JSON

The Python orchestrator:

- Inspects GitHub PR state
- Triggers the reviewer
- Polls for reviewer response
- Collects review comments
- Classifies current vs. stale feedback
- Invokes the coding agent
- Validates the agent's JSON output
- Posts replies and resolves bot threads
- Commits and pushes changes
- Waits for CI via events (not polling)
- Advances the state machine
- Terminates safely

Do not implement the workflow as one giant long-running LLM prompt.

---

## 3. Repository Strategy

Standalone repository:

```
ai-pr-orchestrator/
```

Target repositories contain only:

```
.github/workflows/ai-review-loop.yml
.github/ai-review-loop.yml
AGENTS.md / CLAUDE.md / repository-specific coding instructions
```

### File Structure

Top-level packages only — let internal structure emerge from the code:

```
ai-pr-orchestrator/
  pyproject.toml
  README.md
  AGENTS.md
  examples/
    target-repo-workflow.yml
    sample-config.yml
  src/
    ai_pr_orchestrator/
      __init__.py
      cli.py
      config.py
      models.py
      state_machine.py
      runner.py
      github/
        __init__.py
        client.py
        graphql.py
        models.py
      coders/
        __init__.py
        base.py
        codex_cli.py
      reviewers/
        __init__.py
        base.py
        gemini_github.py
      agents/
        __init__.py
        prompt_builder.py
        output_validator.py
      git/
        __init__.py
        repo.py
      logging.py
      prompts/
        fix_review_threads.md
  tests/
    unit/
    fixtures/
    integration/
  scripts/
    local-run.sh
```

Do not create files until the code needs them. A flat module that grows into a
package is better than a premature package with one function per file.

---

## 4. Product Shape

CLI named `aipro`:

```
aipro run --pr 123
aipro run --event-path "$GITHUB_EVENT_PATH"
aipro dry-run --pr 123
aipro inspect --pr 123
```

Primary production entry point:

```
aipro run --event-path "$GITHUB_EVENT_PATH"
```

Dry-run mode performs zero GitHub mutations.

---

## 5. Configuration

Target repositories configure behavior in `.github/ai-review-loop.yml`:

```yaml
enabled_label: "ai-loop"
done_label: "ai-loop-done"
error_label: "ai-loop-error"

main_coder:
  provider: codex_cli
  command: "codex"
  args:
    - "exec"
    - "{prompt}"
  timeout_seconds: 1800
  output_file: ".ai-orchestrator-result.json"
  env:
    - CODEX_API_KEY

reviewers:
  gemini_github:
    enabled: true
    bot_logins:
      - "gemini-code-assist[bot]"
      - "google-gemini-code-assist[bot]"
    trigger_comment: |
      /gemini review this pull request again. Focus only on correctness,
      security, data loss, race conditions, broken tests, API compatibility,
      and risky behavior changes. Ignore style-only nits unless they hide a
      real defect.

review_phase:
  max_rounds: 1
  poll_interval_seconds: 30
  reviewer_timeout_seconds: 600
  phase_timeout_seconds: 900

thread_policy:
  auto_resolve_bot_threads: true
  never_resolve_human_threads: true
  resolve_rejected_bot_threads: true
  require_reply_before_resolve: true

git:
  base_branch: "main"
  commit_author_name: "AI PR Orchestrator"
  commit_author_email: "ai-pr-orchestrator@example.com"
  commit_message_prefix: "fix: address AI review feedback"

ci:
  require_green_before_done: true
  required_checks: []
  ignored_checks:
    - "AI PR Review Loop"
  relevant_failed_log_lines: 300

safety:
  only_run_on_labeled_prs: true
  disallow_forks: true
  disallow_workflow_file_changes: true
  max_total_iterations: 3
  max_commits_per_run: 1
  max_coder_invocations_per_run: 1
  max_reviewer_triggers_per_run: 3
  max_prompt_tokens: 100000
  allowed_pr_author_associations:
    - "OWNER"
    - "MEMBER"
    - "COLLABORATOR"

notifications:
  mention_on_needs_human:
    - "@pavelkrotkov"
  mention_on_error:
    - "@pavelkrotkov"
```

### Authentication

Provider credentials are injected as environment variables in the GitHub Actions
workflow. The `main_coder.env` list declares which env vars the coder adapter
expects. The orchestrator passes them through to the subprocess and redacts them
from all log output.

```yaml
# In the GitHub Actions workflow:
env:
  GH_TOKEN: ${{ secrets.AI_ORCHESTRATOR_GITHUB_TOKEN }}
  CODEX_API_KEY: ${{ secrets.CODEX_API_KEY }}
```

The orchestrator itself authenticates to GitHub using `GH_TOKEN` /
`GITHUB_TOKEN` via `httpx` (not the `gh` CLI for API calls).

---

## 6. Critical Abstractions

### 6.1 Coder Adapter

```python
class CoderAdapter(Protocol):
    name: str
    def run_fix_task(self, task: FixTask) -> AgentRunResult: ...
```

V1 implements `codex_cli` only. The state machine does not know which coder is
running.

### 6.2 Reviewer Adapter

```python
class ReviewerAdapter(Protocol):
    name: str
    def build_trigger_comment(self, context: ReviewContext) -> str: ...
    def matches_author(self, login: str) -> bool: ...
    def collect_findings(
        self,
        pr: PullRequest,
        since: datetime,
        head_sha: str,
    ) -> list[Finding]: ...
```

V1 implements `gemini_github` only.

### 6.3 Normalized Finding

Every reviewer's feedback is normalized:

```python
@dataclass
class Finding:
    id: str
    source: str
    head_sha: str | None
    thread_id: str | None
    comment_id: str | None
    path: str | None
    line: int | None
    severity: str | None
    body: str
    created_at: datetime
    is_resolved: bool
    is_outdated: bool
    raw: dict
```

Outdated findings (GitHub's `outdated` field) are auto-classified as stale
without invoking the coder.

---

## 7. State Storage

V1 uses a **hidden HTML comment inside a PR comment** as the state store.

```markdown
## AI PR Orchestrator

**Phase:** `review` | **Round:** `1/1` | **Status:** `waiting`
**Head:** `abc1234` | **Updated:** `2026-05-24T14:12:00Z`

<!-- aipro-state
{"version":1,"pr_number":123,"head_sha":"abc123","status":"waiting",...}
-->
```

The orchestrator:

1. Searches for its own status comment (by HTML comment marker)
2. Parses the JSON from the HTML comment
3. Updates the comment with new state + human-readable summary
4. Uses the comment's `updated_at` timestamp as an optimistic concurrency guard

This avoids all git-branch state management complexity. The JSON payload is small
(finding IDs and verdicts, not full diffs or logs).

If the state comment is missing or corrupt, the orchestrator re-initializes from
the PR's current state.

### State Model

```python
@dataclass
class RuntimeState:
    version: int
    pr_number: int
    head_sha: str
    base_sha: str | None
    round_index: int
    status: Literal[
        "init",
        "triggering",
        "waiting",
        "collecting",
        "handling",
        "ci_wait",
        "done",
        "error",
        "needs_human",
    ]
    handled_findings: dict[str, HandledFinding]
    trigger_history: list[ReviewerTrigger]
    cost: CostTracker
    commits_made: list[str]
    created_at: datetime
    updated_at: datetime
    last_error: str | None
    done_reason: str | None
```

---

## 8. State Machine

The state machine is pure. No side effects.

**Input:** current state + PR snapshot + config + current time

**Output:** new state + planned actions

```
if label removed:
    done with reason "label_removed"
if safety checks fail:
    error or needs_human
if cost limits exceeded:
    needs_human with cost summary
if PR head SHA changed since state was saved:
    re-check head SHA before committing coder output (see 8.1)
if status == init:
    move to triggering
if status == triggering:
    trigger reviewer, record trigger, move to waiting
if status == waiting:
    poll for reviewer response or timeout
    collect findings after trigger timestamp
    auto-classify outdated findings as stale
    if no findings: done
    if findings: move to handling
if status == handling:
    build coder task from findings
    invoke coder
    validate JSON
    if tests regress: rollback, needs_human
    apply decisions (reply, resolve bot threads)
    commit/push if worktree changed
    if CI gate enabled: ci_wait and exit
    else: done
if status == ci_wait:
    resume from check_run/check_suite/status events only
    ignore events whose SHA != state.head_sha
    if CI pending: stay ci_wait, exit
    if CI passed: done
    if CI failed: needs_human
if status == done:
    post final summary once, apply done label
if status == needs_human:
    post summary with @-mentions once, apply error label
if status == error:
    post diagnostic once, apply error label
```

### 8.1 Head SHA Race Condition

If someone pushes to the PR branch while the coder is working, the coder's
output is based on stale code. Before committing:

1. Fetch the current remote HEAD of the PR branch
2. If it differs from the HEAD the coder saw, discard coder output
3. Transition to `init` so the next run starts fresh

---

## 9. Planned Action Types

```python
class PlannedAction:
    type: Literal[
        "post_pr_comment",
        "update_status_comment",
        "invoke_coder",
        "reply_to_thread",
        "resolve_thread",
        "commit_changes",
        "push_branch",
        "add_label",
        "remove_label",
        "post_final_summary",
        "noop",
    ]
    payload: dict
```

The executor performs actions in deterministic order and records durable
mutations in state immediately. Reruns skip already-completed actions.

V1 does not force-push, squash, or rewrite history.

---

## 10. Execution Model

The orchestrator runs as a bounded process within a GitHub Actions job.

```python
def run(pr_number: int, config: Config):
    state = load_or_create_state(pr_number)
    while runnable(state.status):
        snapshot = fetch_pr_snapshot(pr_number)
        state, actions = transition(state, snapshot, config, now())
        execute(actions)
        save_state(state)

        if state.status == "waiting":
            poll_until_reviewer_responds_or_timeout(state, config)
        elif state.status == "ci_wait":
            save_state(state)
            return  # exit; CI events will resume
    execute_terminal_actions(state, config)
```

V1 performs at most one coder invocation and one fix commit per job.

CI waiting exits the job. `check_run`, `check_suite`, and `status` events
resume the state machine.

---

## 11. Primary Coding Agent Contract

The coder writes its result to `.ai-orchestrator-result.json`. The orchestrator
deletes the file before invocation, reads it after, and falls back to parsing
stdout if the file is missing.

### Prompt Structure

Each coder adapter owns its prompt formatting. The orchestrator provides
structured data:

```python
@dataclass
class FixTask:
    pr_number: int
    head_sha: str
    base_branch: str
    findings: list[Finding]
    changed_files: list[str]
    diff_text: str
    repo_instructions: str | None
    output_file: str
```

The adapter formats this into a prompt appropriate for its coder (Codex, Claude
Code, etc. have different prompting styles).

### Required Output Schema

```json
{
  "changed": true,
  "commit_message": "string or null",
  "summary": "string",
  "needs_human": false,
  "decisions": [
    {
      "finding_id": "string",
      "thread_id": "string or null",
      "verdict": "accepted|rejected|needs_human",
      "confidence": "low|medium|high",
      "reason": "string",
      "reply": "GitHub reply body",
      "should_resolve": true,
      "changed_files": ["path"]
    }
  ],
  "tests": [
    {
      "command": "string",
      "result": "passed|failed|not_run",
      "notes": "string"
    }
  ],
  "token_usage": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

### Output Validation

The orchestrator validates:

- Valid JSON (file first, stdout fallback)
- Every input finding has exactly one decision
- No unknown finding IDs
- Verdict is from the allowed set
- Every decision has a nonempty reason and reply
- `changed=true` corresponds to actual git changes
- `needs_human=true` is propagated to state

Invalid output moves the run to error.

---

## 12. Decision Application

```
accepted:  reply summarizing fix + resolve bot thread
rejected:  rebuttal with evidence + resolve bot thread if policy allows
needs_human:  explain why + leave unresolved + move state to needs_human
```

Never auto-resolve human-authored threads.

---

## 13. Reviewer Triggering

Each trigger includes a machine marker:

```markdown
<!-- aipro-trigger reviewer=gemini_github round=1 head=abc123 -->
/gemini review this pull request again. ...
```

This prevents duplicate triggering and lets the orchestrator associate responses
with a specific round and HEAD SHA.

---

## 14. GitHub Client

Use `httpx` with the GitHub REST and GraphQL APIs directly. Do not shell out to
`gh` CLI for API calls.

```python
class GitHubClient:
    def get_pr(self, pr_number: int) -> PullRequest: ...
    def get_review_threads(self, pr_number: int) -> list[ReviewThread]: ...
    def post_comment(self, pr_number: int, body: str) -> IssueComment: ...
    def edit_comment(self, comment_id: str, body: str) -> None: ...
    def reply_to_review_thread(self, thread_id: str, body: str) -> None: ...
    def resolve_review_thread(self, thread_id: str) -> None: ...
    def add_label(self, pr_number: int, label: str) -> None: ...
    def remove_label(self, pr_number: int, label: str) -> None: ...
    def get_check_runs(self, ref: str) -> list[CheckRun]: ...
```

Requirements:

- Dry-run mode (no mutations)
- Rate limit handling (respect `Retry-After`, backoff on 403/429)
- Pagination
- Token redaction in logs
- Deterministic fake implementation for tests

---

## 15. Observability

Every state transition and action execution emits a structured JSON log line:

```json
{"ts": "...", "level": "info", "event": "state_transition", "pr": 123,
 "from": "waiting", "to": "handling", "head_sha": "abc123"}
```

All secrets listed in `main_coder.env` are redacted from log output.

The status comment on the PR is the primary human-readable audit trail.

---

## 16. Safety

V1 enforces:

- Require `ai-loop` label (check at every transition, not just init)
- If label removed mid-run, transition to `done` with `done_reason: "label_removed"`
- Skip fork PRs
- Skip untrusted author associations
- Block `.github/workflows/**` changes
- Require clean worktree before invoking coder
- Re-check HEAD SHA before committing (see 8.1)
- Cap max iterations, commits, coder invocations, reviewer triggers
- Redact tokens/secrets from all output
- Do not process orchestrator's own trigger comments as findings
- Do not process comments from old HEAD SHAs as current findings

### Test Regression Rollback

The coder runs targeted tests as part of its task. If the coder reports test
failures, or if the orchestrator detects a dirty worktree with failing tests:

1. Restore the worktree to the pre-coder state (`git checkout .`)
2. Transition to `needs_human` with a summary of what the coder tried

The orchestrator does not re-run the full CI suite — that is what the CI gate is
for. Rollback applies only to the coder's self-reported test results.

### Cost Tracking

```python
@dataclass
class CostTracker:
    coder_invocations: int
    reviewer_triggers: int
    total_api_calls: int
    input_tokens: int
    output_tokens: int
```

Token counts come from the coder's `token_usage` output field. When limits are
hit, transition to `needs_human` with a cost summary.

---

## 17. GitHub Actions Workflow

```yaml
name: AI PR Review Loop

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  issue_comment:
    types: [created]
  pull_request_review:
    types: [submitted]
  pull_request_review_comment:
    types: [created]
  check_run:
    types: [completed]
  check_suite:
    types: [completed]
  status:
  workflow_dispatch:
    inputs:
      pr:
        description: "Pull request number"
        required: true
        type: string

concurrency:
  group: >-
    ${{ github.workflow }}-${{
      github.event.pull_request.number
      || github.event.issue.number
      || github.event.check_run.pull_requests[0].number
      || github.event.check_suite.pull_requests[0].number
      || github.event.inputs.pr
      || github.sha }}
  cancel-in-progress: false

permissions:
  contents: write
  pull-requests: write
  issues: write
  checks: read
  statuses: read

jobs:
  orchestrate:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.AI_ORCHESTRATOR_GITHUB_TOKEN || github.token }}
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install orchestrator
        run: pip install git+https://github.com/YOUR_ORG/ai-pr-orchestrator.git@v0.1.0
      - name: Run orchestrator
        env:
          GH_TOKEN: ${{ secrets.AI_ORCHESTRATOR_GITHUB_TOKEN || github.token }}
          GITHUB_TOKEN: ${{ secrets.AI_ORCHESTRATOR_GITHUB_TOKEN || github.token }}
          CODEX_API_KEY: ${{ secrets.CODEX_API_KEY }}
        run: aipro run --event-path "$GITHUB_EVENT_PATH"
```

CI waiting exits the job. `check_run`/`check_suite`/`status` events wake it.
If events are missed, add a conservative schedule fallback (e.g.,
`cron: "0 */6 * * *"`).

---

## 18. Testing Strategy

### Unit Tests

- Config loading and validation
- State serialization/deserialization
- Pure state transitions (every status path)
- Round advancement and termination
- Finding normalization and outdated auto-classification
- Reviewer author matching
- Prompt building
- Agent JSON validation
- Decision application policy
- Cost tracking and limit enforcement
- Label-removal handling
- Head SHA mismatch detection

### Fake GitHub Client Tests

- State comment create/read/update cycle
- Reviewer trigger posting
- Reviewer response polling
- Finding collection (bot-authored only, ignoring human comments)
- Coder invocation with fake adapter
- Decision application (accept/reject/needs_human)
- Bot thread resolution
- Human thread non-resolution
- Idempotent reruns
- CI event resume from `ci_wait`
- Max rounds termination
- Optimistic concurrency (stale comment detection)

### Fake Agent Tests

- Valid accepted/rejected/needs_human findings
- Invalid JSON
- Missing decisions, unknown finding IDs
- Output file missing (stdout fallback)
- Timeout and nonzero exit

### Local Git Tests

- Clean/dirty worktree detection
- Commit when changed, no commit when unchanged
- One commit max
- Test regression rollback
- Push (mocked)
- No force-push or squash

### Live GitHub Tests

Optional, behind explicit env vars. Never run by default.

---

## 19. Implementation Order

1. Package skeleton and CLI
2. Config loader
3. Core models and state (PR comment store)
4. Pure state machine with fake data
5. Fake GitHub client
6. Reviewer adapter interface + Gemini adapter
7. Coder adapter interface + Codex CLI adapter
8. Prompt builder and JSON validator
9. Decision application
10. GitHub client (httpx, REST + GraphQL)
11. Git manager
12. Runner loop with reviewer polling and CI suspension
13. Event-path parsing and CI resume
14. Structured logging
15. Dry-run
16. End-to-end fake-client happy path
17. Real GitHub dry-run
18. GitHub Actions workflow

Do not start by wiring real Codex or Gemini into GitHub mutations. Get the pure
state machine and fake-client tests passing first.

---

## 20. V1 Definition of Done

- `aipro dry-run --pr N` shows state and planned actions with zero mutations
- `aipro run --pr N` creates/reads/updates state via PR comment
- Gemini reviewer can be triggered and findings collected
- Outdated findings auto-classified as stale
- Fake coder returns decisions that are applied to GitHub threads
- Bot threads replied to and resolved; human threads never auto-resolved
- One commit per run; no commit when unchanged
- CI gate works: `ci_wait` exits the job, check events resume it
- Test regression triggers rollback
- Label removal transitions to done
- Head SHA mismatch discards stale coder output
- Cost limits enforced (invocation counts + token counts)
- Safety checks pass (forks, labels, author associations, workflow files)
- Structured JSON logging with secret redaction
- Unit tests cover the state machine
- Fake-client tests cover the full happy path
- Rate limits handled with backoff

---

## 21. V1 Non-Goals

See [plan-v2.md](plan-v2.md) for everything deferred from V1.
