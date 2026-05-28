"""Orchestrator runner: executes the state machine loop and side effects.

The Runner glues together the pure state machine, GitHub I/O, the coder
adapter, reviewer adapters, and the git worktree. State is persisted in a
single hidden PR comment between invocations.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_pr_orchestrator.coders.base import CoderAdapter
from ai_pr_orchestrator.config import Config, load_config
from ai_pr_orchestrator.git.repo import GitRepo
from ai_pr_orchestrator.github import models as gh_models
from ai_pr_orchestrator.github.protocol import GitHubClient
from ai_pr_orchestrator.models import (
    AgentRunResult,
    CheckRun,
    Finding,
    FixTask,
    PlannedAction,
    PullRequest,
    ReviewerTrigger,
    RuntimeState,
)
from ai_pr_orchestrator.reviewers.base import ReviewerAdapter
from ai_pr_orchestrator.state_machine import TERMINAL_STATUSES, TransitionSnapshot, transition
from ai_pr_orchestrator.state_storage import (
    find_state_comment,
    serialize_state_comment,
)

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED_MESSAGE = "AI PR Orchestrator runner is not implemented yet."

# Hard cap on transitions per Runner.run to prevent pathological infinite loops
# if a future state machine bug fails to converge. Real runs should terminate
# in far fewer iterations.
_MAX_TRANSITIONS_PER_RUN = 50

# Hard cap on reviewer poll iterations as a guardrail against a non-advancing
# clock (or extreme clock skew) that would otherwise leave the poll loop
# unable to detect timeouts. The cap is independent of timeout-based exits
# and is a defense-in-depth measure. When tripped, the loop treats the
# situation as a reviewer timeout so the state machine routes to
# ``needs_human`` rather than spinning forever.
_MAX_POLL_ITERATIONS = 1000


@dataclass(frozen=True)
class ParsedEvent:
    event_type: str
    pr_number: int | None
    head_sha: str | None


@dataclass
class RunnerContext:
    github: GitHubClient
    coder: CoderAdapter
    reviewers: dict[str, ReviewerAdapter]
    config: Config
    git: GitRepo | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    sleeper: Callable[[float], None] = field(default=time.sleep)


def parse_event(event: Any, *, event_name: str | None = None) -> ParsedEvent:
    """Parse a GitHub Actions event payload into a typed summary.

    The payload itself does not include the event type; callers may pass
    ``event_name`` (typically from ``GITHUB_EVENT_NAME``) to disambiguate.
    Without a hint, we infer from the keys present in the payload.

    Note on ``status`` events: GitHub's status webhook carries only a commit
    SHA (no PR number). Mapping SHA -> PR requires a live GitHub client
    (``GET /repos/{owner}/{repo}/commits/{sha}/pulls``), which the CLI does
    not currently wire up. The CLI surfaces a clear error in that case and
    asks the operator to pass ``--pr`` explicitly. ``parsed.head_sha`` is
    still populated so downstream consumers (e.g. a future ``status``-aware
    CLI) can do the lookup.
    """
    if not isinstance(event, dict):
        # Malformed payloads (lists, scalars, null) can't yield a PR; bail safely.
        return ParsedEvent(event_type=event_name or "unknown", pr_number=None, head_sha=None)
    inferred = event_name or _infer_event_name(event)
    if inferred == "pull_request" or inferred == "pull_request_review":
        pr = event.get("pull_request") or {}
        return ParsedEvent(
            event_type=inferred,
            pr_number=_safe_int(pr.get("number")),
            head_sha=_safe_str((pr.get("head") or {}).get("sha")),
        )
    if inferred == "issue_comment":
        issue = event.get("issue") or {}
        pr_link = issue.get("pull_request")
        pr_number = _safe_int(issue.get("number")) if pr_link else None
        return ParsedEvent(event_type=inferred, pr_number=pr_number, head_sha=None)
    if inferred == "check_run":
        cr = event.get("check_run") or {}
        prs = cr.get("pull_requests") or []
        first = prs[0] if prs else {}
        return ParsedEvent(
            event_type=inferred,
            pr_number=_safe_int(first.get("number")),
            head_sha=_safe_str(cr.get("head_sha")),
        )
    if inferred == "check_suite":
        cs = event.get("check_suite") or {}
        prs = cs.get("pull_requests") or []
        first = prs[0] if prs else {}
        return ParsedEvent(
            event_type=inferred,
            pr_number=_safe_int(first.get("number")),
            head_sha=_safe_str(cs.get("head_sha")),
        )
    if inferred == "status":
        return ParsedEvent(
            event_type=inferred,
            pr_number=None,
            head_sha=_safe_str(event.get("sha")),
        )
    if inferred == "workflow_dispatch":
        inputs = event.get("inputs") or {}
        raw = inputs.get("pr")
        pr_number: int | None
        try:
            pr_number = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            pr_number = None
        return ParsedEvent(event_type=inferred, pr_number=pr_number, head_sha=None)
    return ParsedEvent(event_type=inferred or "unknown", pr_number=None, head_sha=None)


def _infer_event_name(event: dict[str, Any]) -> str:
    if "check_run" in event:
        return "check_run"
    if "check_suite" in event:
        return "check_suite"
    if "inputs" in event and (event["inputs"] or {}).get("pr") is not None:
        return "workflow_dispatch"
    if "issue" in event and "pull_request" in (event.get("issue") or {}):
        return "issue_comment"
    if "review" in event and "pull_request" in event:
        return "pull_request_review"
    if "pull_request" in event:
        return "pull_request"
    if "sha" in event and "state" in event:
        return "status"
    return "unknown"


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _safe_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


class Runner:
    """Drive one orchestration cycle for a single PR."""

    def __init__(self, ctx: RunnerContext) -> None:
        self._ctx = ctx
        # Cached per-run coder output; consumed by the next transition.
        self._pending_coder_result: AgentRunResult | None = None
        # Track findings collected this cycle so invoke_coder can build a FixTask.
        self._current_findings: list[Finding] = []
        # Track whether we already produced a commit this run as a safety net.
        self._commits_this_run = 0
        # ID of the hidden PR comment that persists RuntimeState. Resolved once
        # by ``_load_or_init_state`` and reused by every ``_save_state`` call
        # within the run. Stored on the instance so action handlers (e.g.
        # ``commit_changes``) can checkpoint state without threading the id
        # through every helper.
        self._state_comment_id: int | None = None

    def run(self, pr_number: int, *, event: ParsedEvent | None = None) -> int:
        try:
            return self._run(pr_number, event)
        except Exception:
            logger.exception("Runner crashed for PR #%s", pr_number)
            return 1

    def _run(self, pr_number: int, event: ParsedEvent | None) -> int:
        ctx = self._ctx
        gh_pr = ctx.github.get_pr(pr_number)
        state = self._load_or_init_state(pr_number, gh_pr)

        # Push recovery: if a previous run committed locally but the process
        # died before push_branch ran (or push itself failed mid-flight), the
        # checkpoint we saved after commit_changes shows a commit in
        # commits_made — but the remote branch is still on the prior SHA. On
        # resume, detect that and reissue the push so the worktree and the
        # remote converge. Failures are logged but do not abort the run; the
        # main state machine will eventually surface a hard error if push is
        # genuinely impossible.
        if ctx.git is not None and state.commits_made:
            try:
                local_head = ctx.git.get_head_sha()
                if local_head == state.commits_made[-1]:
                    remote_head = ctx.git.fetch_remote_head(gh_pr.head_ref)
                    if remote_head != local_head:
                        logger.info(
                            "Local commit %s not on remote; resuming push for PR #%s",
                            local_head,
                            pr_number,
                        )
                        ctx.git.push(gh_pr.head_ref)
            except Exception:
                logger.exception("Push recovery failed for PR #%s", pr_number)

        # Detect an orphaned coder invocation from a prior process. If the
        # state machine had asked us to invoke the coder (last_coder_round_index
        # == round_index) but our in-memory _pending_coder_result is empty
        # because this is a fresh Runner instance, the prior coder result is
        # gone for good. Without intervention the loop would emit
        # ``noop(waiting_for_coder)`` forever. Bail to ``needs_human`` so a
        # human can decide whether to retry.
        if (
            state.status == "handling"
            and state.last_coder_round_index == state.round_index
            and self._pending_coder_result is None
        ):
            now = ctx.clock()
            state = replace(
                state,
                status="needs_human",
                last_error="coder_invocation_orphaned",
                updated_at=now,
            )
            snapshot = self._build_snapshot(state, gh_pr, event=event)
            _, actions = transition(state, snapshot, ctx.config, now)
            state = self._execute_actions(actions, pr_number, gh_pr, state)
            self._save_state(pr_number, state)
            return 0

        for _ in range(_MAX_TRANSITIONS_PER_RUN):
            if state.status in TERMINAL_STATUSES:
                self._save_state(pr_number, state)
                return 0

            # waiting/collecting need reviewer polling with a clock-aware loop.
            if state.status in ("waiting", "collecting"):
                state = self._poll_reviewers(state, pr_number, gh_pr)
                continue

            # ci_wait: do one transition with current checks; if still pending, exit.
            if state.status == "ci_wait":
                snapshot = self._build_snapshot(state, gh_pr, event=event)
                next_state, actions = transition(state, snapshot, ctx.config, ctx.clock())
                if next_state.status == "ci_wait":
                    # CI hasn't completed yet; persist state and let the next
                    # check_run/check_suite event wake us back up.
                    self._save_state(pr_number, next_state)
                    return 0
                state = self._execute_actions(actions, pr_number, gh_pr, next_state)
                self._save_state(pr_number, state)
                continue

            snapshot = self._build_snapshot(state, gh_pr, event=event)
            next_state, actions = transition(state, snapshot, ctx.config, ctx.clock())
            # Clear the cached coder result once the transition has consumed it
            # so a subsequent transition in the same run doesn't see a stale one.
            self._pending_coder_result = None
            state = self._execute_actions(actions, pr_number, gh_pr, next_state)
            self._save_state(pr_number, state)

        logger.error("Runner exceeded max transitions for PR #%s", pr_number)
        return 1

    # ---- State loading / saving ----

    def _load_or_init_state(self, pr_number: int, gh_pr: gh_models.PullRequest) -> RuntimeState:
        comments = self._ctx.github.get_pr_comments(pr_number)
        sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
        if sc is not None:
            self._state_comment_id = int(sc.comment_id) if isinstance(sc.comment_id, int) else None
            return sc.state

        now = self._ctx.clock()
        state = RuntimeState(
            version=1,
            pr_number=pr_number,
            head_sha=gh_pr.head_sha,
            status="init",
            created_at=now,
            updated_at=now,
        )
        posted = self._ctx.github.post_comment(pr_number, serialize_state_comment(state))
        self._state_comment_id = posted.id
        return state

    def _save_state(self, pr_number: int, state: RuntimeState) -> None:
        body = serialize_state_comment(state)
        if self._state_comment_id is None:
            posted = self._ctx.github.post_comment(pr_number, body)
            self._state_comment_id = posted.id
            return
        self._ctx.github.edit_comment(self._state_comment_id, body)

    # ---- Snapshot builders ----

    def _build_snapshot(
        self,
        state: RuntimeState,
        gh_pr: gh_models.PullRequest,
        *,
        event: ParsedEvent | None,
        reviewer_timed_out: bool = False,
        reviewer_responded: bool = False,
    ) -> TransitionSnapshot:
        ctx = self._ctx
        pr = _convert_pr(gh_pr, ctx.config)
        findings: list[Finding] | None = None
        checks: list[CheckRun] | None = None
        coder_result = self._pending_coder_result
        worktree_changed = False
        remote_head_sha: str | None = None
        event_head_sha = event.head_sha if event else None

        if state.status in ("waiting", "collecting"):
            findings = self._collect_findings(state, gh_pr)
            self._current_findings = list(findings)

        if state.status == "handling":
            # The state machine consumes _current_findings via the snapshot to
            # build FixTask payloads; reload them if not already populated.
            if not self._current_findings:
                self._current_findings = self._collect_findings(state, gh_pr)
            findings = list(self._current_findings)
            if ctx.git is not None:
                worktree_changed = not ctx.git.is_clean()
                try:
                    remote_head_sha = ctx.git.fetch_remote_head(gh_pr.head_ref)
                except Exception:
                    logger.exception("Failed to fetch remote head for ref %s", gh_pr.head_ref)
                    remote_head_sha = None

        if state.status == "ci_wait":
            raw_checks = ctx.github.get_check_runs(state.head_sha)
            checks = [_convert_check_run(cr, state.head_sha) for cr in raw_checks]

        return TransitionSnapshot(
            pr=pr,
            findings=findings,
            checks=checks,
            coder_result=coder_result,
            remote_head_sha=remote_head_sha,
            event_head_sha=event_head_sha,
            reviewer_responded=reviewer_responded,
            reviewer_timed_out=reviewer_timed_out,
            worktree_changed=worktree_changed,
        )

    def _collect_findings(self, state: RuntimeState, gh_pr: gh_models.PullRequest) -> list[Finding]:
        ctx = self._ctx
        all_findings: list[Finding] = []
        triggers_by_reviewer: dict[str, ReviewerTrigger] = {}
        for trig in state.trigger_history:
            existing = triggers_by_reviewer.get(trig.reviewer_name)
            if existing is None or trig.timestamp > existing.timestamp:
                triggers_by_reviewer[trig.reviewer_name] = trig

        for name, reviewer in ctx.reviewers.items():
            cfg = ctx.config.reviewers.get(name)
            if cfg is not None and not cfg.enabled:
                continue
            trig = triggers_by_reviewer.get(name)
            ts = trig.timestamp if trig is not None else state.created_at
            try:
                found = reviewer.collect_findings(state.pr_number, gh_pr.head_sha, ts)
            except Exception:
                logger.exception("Reviewer %s failed to collect findings", name)
                continue
            all_findings.extend(found)
        return all_findings

    # ---- Reviewer polling ----

    def _poll_reviewers(
        self,
        state: RuntimeState,
        pr_number: int,
        gh_pr: gh_models.PullRequest,
    ) -> RuntimeState:
        ctx = self._ctx
        cfg = ctx.config.review_phase
        # Deadlines are anchored to persisted state, NOT to ctx.clock() at
        # resume time. A new webhook arriving 3h after the trigger must observe
        # the same deadline as the original invocation — otherwise an already
        # timed-out ``waiting`` state would keep getting a fresh budget on
        # every wake-up and never route to ``needs_human``.
        #
        # reviewer_timeout: anchored to the most recent ReviewerTrigger (the
        # newest reviewer the runner is waiting on). Fallback: state.created_at.
        # phase_timeout: anchored to the *first* ReviewerTrigger so the overall
        # phase budget caps total wall time across multiple reviewer rounds.
        # Fallback: state.created_at.
        if state.trigger_history:
            latest_trigger_ts = max(t.timestamp for t in state.trigger_history)
            earliest_trigger_ts = min(t.timestamp for t in state.trigger_history)
        else:
            latest_trigger_ts = state.created_at
            earliest_trigger_ts = state.created_at
        deadline_reviewer = latest_trigger_ts + timedelta(seconds=cfg.reviewer_timeout_seconds)
        deadline_phase = earliest_trigger_ts + timedelta(seconds=cfg.phase_timeout_seconds)

        for _iteration in range(_MAX_POLL_ITERATIONS):
            snapshot = self._build_snapshot(state, gh_pr, event=None)
            if snapshot.findings:
                next_state, actions = transition(state, snapshot, ctx.config, ctx.clock())
                state = next_state
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
                return state

            # If the reviewer responded but produced zero findings, treat the
            # phase as complete and let the state machine route to
            # ``done``/``no_findings`` via ``reviewer_responded=True``.
            if self._all_enabled_reviewers_responded(state, gh_pr):
                responded_snapshot = self._build_snapshot(
                    state, gh_pr, event=None, reviewer_responded=True
                )
                next_state, actions = transition(state, responded_snapshot, ctx.config, ctx.clock())
                state = next_state
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
                return state

            now = ctx.clock()
            if now >= deadline_reviewer or now >= deadline_phase:
                timed_out_snapshot = self._build_snapshot(
                    state, gh_pr, event=None, reviewer_timed_out=True
                )
                next_state, actions = transition(state, timed_out_snapshot, ctx.config, ctx.clock())
                state = next_state
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
                return state

            ctx.sleeper(cfg.poll_interval_seconds)

        # Guardrail: a non-advancing clock or pathological config left the
        # loop unable to detect timeouts. Treat as a reviewer timeout so the
        # state machine routes to ``needs_human``.
        logger.warning(
            "Reviewer poll loop hit hard iteration cap (%d) for PR #%s; "
            "treating as reviewer timeout",
            _MAX_POLL_ITERATIONS,
            pr_number,
        )
        timed_out_snapshot = self._build_snapshot(state, gh_pr, event=None, reviewer_timed_out=True)
        next_state, actions = transition(state, timed_out_snapshot, ctx.config, ctx.clock())
        state = next_state
        state = self._execute_actions(actions, pr_number, gh_pr, state)
        self._save_state(pr_number, state)
        return state

    def _all_enabled_reviewers_responded(
        self, state: RuntimeState, gh_pr: gh_models.PullRequest
    ) -> bool:
        """Return True iff every enabled reviewer has posted *something* after
        its trigger. Used to detect the zero-findings completion case so the
        state machine can transition to ``done`` instead of timing out.
        """
        ctx = self._ctx
        triggers_by_reviewer: dict[str, ReviewerTrigger] = {}
        for trig in state.trigger_history:
            existing = triggers_by_reviewer.get(trig.reviewer_name)
            if existing is None or trig.timestamp > existing.timestamp:
                triggers_by_reviewer[trig.reviewer_name] = trig

        enabled_names = [
            name
            for name in ctx.reviewers
            if (cfg := ctx.config.reviewers.get(name)) is None or cfg.enabled
        ]
        if not enabled_names:
            return False

        for name in enabled_names:
            reviewer = ctx.reviewers[name]
            has_responded = getattr(reviewer, "has_responded", None)
            if not callable(has_responded):
                # Reviewer adapter doesn't implement the optional probe; we
                # cannot prove a response, so refuse to short-circuit.
                return False
            trig = triggers_by_reviewer.get(name)
            ts = trig.timestamp if trig is not None else state.created_at
            try:
                if not has_responded(state.pr_number, ts):
                    return False
            except Exception:
                logger.exception("Reviewer %s has_responded() failed; assuming no response", name)
                return False
        return True

    # ---- Action execution ----

    def _execute_actions(
        self,
        actions: list[PlannedAction],
        pr_number: int,
        gh_pr: gh_models.PullRequest,
        state: RuntimeState,
    ) -> RuntimeState:
        for action in actions:
            state = self._execute_action(action, pr_number, gh_pr, state)
        return state

    def _execute_action(
        self,
        action: PlannedAction,
        pr_number: int,
        gh_pr: gh_models.PullRequest,
        state: RuntimeState,
    ) -> RuntimeState:
        ctx = self._ctx
        payload = action.payload
        kind = action.type

        if kind == "noop":
            return state
        if kind == "post_pr_comment":
            ctx.github.post_comment(pr_number, str(payload.get("body", "")))
            return state
        if kind == "update_status_comment":
            # The save_state call after action execution writes the canonical
            # state. This action is a hint; no extra side effect needed here.
            return state
        if kind == "reply_to_thread":
            thread_id = payload.get("thread_id")
            if isinstance(thread_id, str):
                ctx.github.reply_to_review_thread(thread_id, str(payload.get("body", "")))
            else:
                logger.warning("reply_to_thread action missing string thread_id; skipping")
            return state
        if kind == "resolve_thread":
            thread_id = payload.get("thread_id")
            if isinstance(thread_id, str):
                ctx.github.resolve_review_thread(thread_id)
            else:
                logger.warning("resolve_thread action missing string thread_id; skipping")
            return state
        if kind == "add_label":
            label = payload.get("label")
            if isinstance(label, str):
                ctx.github.add_label(pr_number, label)
            else:
                logger.warning("add_label action missing string label; skipping")
            return state
        if kind == "remove_label":
            label = payload.get("label")
            if isinstance(label, str):
                ctx.github.remove_label(pr_number, label)
            else:
                logger.warning("remove_label action missing string label; skipping")
            return state
        if kind == "post_final_summary":
            ctx.github.post_comment(pr_number, _build_final_summary(state, payload))
            return state
        if kind == "rollback_changes":
            if ctx.git is not None:
                ctx.git.rollback()
            return state
        if kind == "commit_changes":
            new_state = self._do_commit(payload, state)
            # Checkpoint immediately so a crash (or push failure) between
            # commit and push doesn't leave a local commit invisible to the
            # next run. Without this, ``state.commits_made`` would still be
            # empty after a successful commit, and on resume the runner would
            # have no way to detect that it owes the remote a push. With the
            # checkpoint, the next run sees the local commit recorded and the
            # push-recovery branch at the top of ``_run`` reissues the push.
            if new_state is not state:
                self._save_state(pr_number, new_state)
            return new_state
        if kind == "push_branch":
            self._do_push(gh_pr)
            return state
        if kind == "invoke_coder":
            self._do_invoke_coder(payload, gh_pr, state)
            return state

        logger.warning("Unknown action type: %s", kind)
        return state

    def _do_commit(self, payload: dict[str, Any], state: RuntimeState) -> RuntimeState:
        ctx = self._ctx
        if ctx.git is None:
            return state
        # Safety net: never commit twice within a single Runner.run invocation.
        # Cumulative state.commits_made is not checked here because subsequent
        # rounds (max_total_iterations > 1) legitimately add new commits.
        if self._commits_this_run >= 1:
            return state
        message = str(payload.get("message") or ctx.config.git.commit_message_prefix)
        sha = ctx.git.commit(
            message,
            ctx.config.git.commit_author_name,
            ctx.config.git.commit_author_email,
        )
        if sha is None:
            return state
        self._commits_this_run += 1
        # Track the new commit SHA on state so downstream consumers (e.g. the
        # CI gate's get_check_runs(state.head_sha)) see the just-pushed commit
        # instead of the pre-commit HEAD.
        return replace(state, head_sha=sha, commits_made=[*state.commits_made, sha])

    def _do_push(self, gh_pr: gh_models.PullRequest) -> None:
        ctx = self._ctx
        if ctx.git is None:
            return
        branch = gh_pr.head_ref or ctx.config.git.base_branch
        ctx.git.push(branch)

    def _do_invoke_coder(
        self,
        payload: dict[str, Any],
        gh_pr: gh_models.PullRequest,
        state: RuntimeState,
    ) -> None:
        ctx = self._ctx
        finding_ids = payload.get("finding_ids") or []
        findings_by_id = {f.id: f for f in self._current_findings}
        selected = [findings_by_id[fid] for fid in finding_ids if fid in findings_by_id]
        base_branch = gh_pr.base_ref or ctx.config.git.base_branch
        diff_text = ""
        if ctx.git is not None:
            try:
                diff_text = ctx.git.get_diff(base_branch)
            except Exception:
                logger.exception("Failed to compute diff against base branch %s", base_branch)
                diff_text = ""
        task = FixTask(
            pr_number=state.pr_number,
            head_sha=state.head_sha,
            base_branch=base_branch,
            findings=selected,
            changed_files=list(gh_pr.changed_files),
            diff_text=diff_text,
            output_file=ctx.config.main_coder.output_file,
        )
        result = ctx.coder.run_fix_task(task)
        self._pending_coder_result = result


# ---- Pure helpers ----


def _convert_pr(gh_pr: gh_models.PullRequest, config: Config) -> PullRequest:
    """Adapt a GitHub-API-shaped PullRequest into the orchestrator's model.

    ``author_association`` is propagated from the GitHub API response so the
    state machine's ``_safety_transition`` check (``author_association not in
    allowed_pr_author_associations``) gates real untrusted authors. If the
    field is missing (older test fixtures, dry-runs against payloads that
    omit it), we fall back to the sentinel ``"NONE"`` — which is GitHub's
    own value for "no association" and is treated as untrusted by the
    default allowlist (``OWNER``/``MEMBER``/``COLLABORATOR``).

    ``is_fork`` and ``changed_files`` are propagated from the GitHub model
    so the state machine's ``disallow_forks`` and
    ``disallow_workflow_file_changes`` safety checks are accurate.
    """
    # ``config`` is accepted for forward compatibility with other adapter
    # decisions but is intentionally not used to derive author_association,
    # which must come from the real PR payload.
    del config
    return PullRequest(
        number=gh_pr.number,
        head_sha=gh_pr.head_sha,
        base_sha=gh_pr.base_ref,
        title=gh_pr.title,
        author_login=gh_pr.author,
        author_association=gh_pr.author_association or "NONE",
        labels=list(gh_pr.labels),
        is_draft=gh_pr.draft,
        is_fork=gh_pr.is_fork,
        changed_files=list(gh_pr.changed_files),
    )


def _convert_check_run(cr: gh_models.CheckRun, head_sha: str) -> CheckRun:
    return CheckRun(
        id=str(cr.id),
        name=cr.name,
        status=cr.status,
        conclusion=cr.conclusion,
        head_sha=head_sha,
    )


def _build_final_summary(state: RuntimeState, payload: dict[str, Any]) -> str:
    reason = payload.get("reason") or state.done_reason or state.last_error or "completed"
    cost = state.cost
    return (
        f"AI PR Orchestrator: status `{state.status}` — reason: `{reason}`. "
        f"Coder invocations: {cost.coder_invocations}, "
        f"reviewer triggers: {cost.reviewer_triggers}, "
        f"tokens: {cost.input_tokens + cost.output_tokens}."
    )


# ---- CLI entry points ----


def run(*, pr_number: int, dry_run: bool, event_path: Path | None = None) -> int:
    """Run the orchestrator for a pull request.

    Builds dependencies from configuration and the environment, then delegates
    to ``Runner.run``. ``dry_run`` short-circuits to a no-op for now (V1).
    """
    if dry_run:
        print("Dry-run mode is not yet wired through Runner; exiting cleanly.", file=sys.stderr)
        return 0

    event: ParsedEvent | None = None
    if event_path is not None:
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"Failed to read event file {event_path}: {exc}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"Event file {event_path} is not valid JSON: {exc}", file=sys.stderr)
            return 1
        event = parse_event(payload, event_name=os.environ.get("GITHUB_EVENT_NAME"))

    try:
        config = load_config()
    except Exception as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    ctx = _build_runtime_context(config)
    if ctx is None:
        return 1
    return Runner(ctx).run(pr_number, event=event)


def inspect(*, pr_number: int) -> int:
    """Print the current orchestrator state for a pull request."""
    try:
        config = load_config()
    except Exception as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1
    ctx = _build_runtime_context(config)
    if ctx is None:
        return 1
    comments = ctx.github.get_pr_comments(pr_number)
    sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
    if sc is None:
        print(json.dumps({"pr_number": pr_number, "status": "uninitialized"}))
        return 0
    print(json.dumps(sc.state.to_dict(), indent=2, sort_keys=True))
    return 0


def _build_runtime_context(_config: Config) -> RunnerContext | None:
    # Real construction of GitHub client / coder / reviewers from environment
    # and config is left to a follow-up issue. The Runner itself is fully
    # exercised by unit tests via injected fakes.
    print(NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return None
