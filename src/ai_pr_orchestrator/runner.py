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
from ai_pr_orchestrator.logging import (
    collect_secret_values,
    log_action,
    log_error,
    log_state_transition,
    setup_logging,
)
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
from ai_pr_orchestrator.state_machine import (
    TERMINAL_STATUSES,
    TransitionSnapshot,
    terminal_actions,
    transition,
)
from ai_pr_orchestrator.state_storage import (
    StateComment,
    StateConflictError,
    find_state_comment,
    lock_timestamp,
    prepare_state_comment_update,
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
    # When True the runner plans a single transition and reports the actions it
    # *would* take without performing any GitHub or git mutation. The GitHub
    # client should also be constructed in dry-run mode (defense in depth), and
    # ``git`` should be ``None`` so no commit/push path is reachable.
    dry_run: bool = False


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
        pr = _as_dict(event.get("pull_request"))
        head = _as_dict(pr.get("head"))
        return ParsedEvent(
            event_type=inferred,
            pr_number=_safe_int(pr.get("number")),
            head_sha=_safe_str(head.get("sha")),
        )
    if inferred == "issue_comment":
        issue = _as_dict(event.get("issue"))
        pr_link = issue.get("pull_request")
        pr_number = _safe_int(issue.get("number")) if pr_link else None
        return ParsedEvent(event_type=inferred, pr_number=pr_number, head_sha=None)
    if inferred == "check_run":
        cr = _as_dict(event.get("check_run"))
        first = _first_dict(cr.get("pull_requests"))
        return ParsedEvent(
            event_type=inferred,
            pr_number=_safe_int(first.get("number")),
            head_sha=_safe_str(cr.get("head_sha")),
        )
    if inferred == "check_suite":
        cs = _as_dict(event.get("check_suite"))
        first = _first_dict(cs.get("pull_requests"))
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
        inputs = _as_dict(event.get("inputs"))
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
    if "inputs" in event and _as_dict(event.get("inputs")).get("pr") is not None:
        return "workflow_dispatch"
    if "issue" in event and "pull_request" in _as_dict(event.get("issue")):
        return "issue_comment"
    if "review" in event and "pull_request" in event:
        return "pull_request_review"
    if "pull_request" in event:
        return "pull_request"
    if "sha" in event and "state" in event:
        return "status"
    return "unknown"


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else an empty dict.

    GitHub event payloads are normally well-formed, but a malformed or mocked
    payload could carry a non-dict where we expect a nested object (e.g.
    ``inputs`` as a string). Coercing to ``{}`` lets the ``.get()`` chains
    below stay total instead of raising ``AttributeError``.
    """
    return value if isinstance(value, dict) else {}


def _first_dict(value: Any) -> dict[str, Any]:
    """Return the first element of ``value`` if it is a non-empty list of
    dicts, else an empty dict. Guards the ``pull_requests[0]`` access in
    ``check_run``/``check_suite`` payloads."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


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
        # Optimistic-concurrency lock: the ``updated_at`` of the state we last
        # loaded or saved. ``_save_state`` re-fetches the comment and refuses
        # to overwrite if its ``updated_at`` differs from this expectation,
        # which prevents two concurrent webhook runs from blindly clobbering
        # each other's mutations.
        self._state_expected_updated_at: datetime | None = None

    def run(self, pr_number: int, *, event: ParsedEvent | None = None) -> int:
        try:
            return self._run(pr_number, event)
        except StateConflictError:
            # Another concurrent runner mutated the state comment in the
            # window between our load and save. Their work is authoritative;
            # discard ours and exit cleanly. The next webhook event will pick
            # up wherever they left off.
            logger.warning(
                "State conflict on PR #%s; another runner advanced state first", pr_number
            )
            return 0
        except Exception as exc:
            log_error(logger, error=f"Runner crashed for PR #{pr_number}: {exc}", pr=pr_number)
            return 1

    def _run(self, pr_number: int, event: ParsedEvent | None) -> int:
        ctx = self._ctx
        gh_pr = ctx.github.get_pr(pr_number)
        state = self._load_or_init_state(pr_number, gh_pr)

        # Already terminal: the previous run persisted a final state and emitted
        # its side effects. Bail before any setup (push recovery, orphaned-coder
        # check, transition loop) so a stray webhook on a finished PR is a cheap
        # no-op rather than another full pass with redundant get_pr calls.
        if state.status in TERMINAL_STATUSES:
            return 0

        # Dry-run: plan a single transition from the loaded state and report the
        # actions that would fire, performing zero mutations. We branch here —
        # before the fork hard-block, push recovery, and the transition loop —
        # because those paths exist to *drive* side effects (push, label, final
        # summary) that dry-run must not perform.
        if ctx.dry_run:
            return self._plan_dry_run(pr_number, gh_pr, state, event)

        # Fork hard-block: when git is wired (the commit/push path is live) we
        # cannot safely handle a fork PR. ``gh_pr.head_ref`` is just the branch
        # name on the contributor's fork; ``git push origin
        # HEAD:refs/heads/{head_ref}`` would create/update that branch in the
        # *base* repo instead of the fork's PR head — silently diverging from
        # the PR and possibly clobbering a base-repo branch. Pushing to the fork
        # remote isn't wired yet, so route to a terminal ``needs_human`` with a
        # clear reason rather than corrupting the base repo. (Review-only/
        # dry-run mode runs with ``ctx.git is None`` and is unaffected, so fork
        # *review* still works; only AI auto-fixes are blocked.) Note this is
        # distinct from ``safety.disallow_forks``, which errors out forks
        # entirely; here forks were explicitly allowed for review but still
        # can't be pushed to.
        #
        # Only fire when the orchestrator is actually enabled for this PR. With
        # the default ``only_run_on_labeled_prs=True``, a webhook for an
        # *unlabeled* fork PR must be a no-op — posting a ``needs_human`` summary
        # and adding ``ai-loop-error`` for a PR we were never enabled on would
        # be noise. When ``only_run_on_labeled_prs`` is False, every PR is in
        # scope so the gate is vacuously satisfied.
        orchestrator_enabled = (
            not ctx.config.safety.only_run_on_labeled_prs
            or ctx.config.enabled_label in gh_pr.labels
        )
        if ctx.git is not None and gh_pr.is_fork and orchestrator_enabled:
            logger.warning(
                "Fork PR #%s cannot be pushed to (fork remotes unsupported); needs_human",
                pr_number,
            )
            now = ctx.clock()
            state = replace(
                state,
                status="needs_human",
                last_error="fork_pr_push_unsupported",
                updated_at=now,
            )
            actions = terminal_actions(state, ctx.config)
            state = self._execute_actions(actions, pr_number, gh_pr, state)
            self._save_state(pr_number, state)
            return 0

        # Push recovery: if a previous run committed locally but the process
        # died before push_branch ran (or push itself failed mid-flight), the
        # checkpoint we saved after commit_changes shows a commit in
        # commits_made — but the remote branch is still on the prior SHA. On
        # resume, detect that and reissue the push so the worktree and the
        # remote converge. If the recovery push raises, do NOT fall through:
        # the persisted state would otherwise sit in ci_wait forever waiting
        # for check_run events that can never arrive for an unpushed commit.
        # Route to a terminal ``needs_human`` so the operator sees the real
        # blocker.
        if ctx.git is not None and state.commits_made:
            try:
                # Guard before touching git: an empty head_ref would make
                # fetch_remote_head/push operate on a bogus ref. Raise so the
                # except-clause below routes to a terminal needs_human instead
                # of issuing git commands against the base branch or "".
                if not gh_pr.head_ref:
                    raise ValueError(f"Cannot push: head_ref is empty for PR #{pr_number}")
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
                now = ctx.clock()
                state = replace(
                    state,
                    status="needs_human",
                    last_error="push_recovery_failed",
                    updated_at=now,
                )
                # transition() returns no actions for an already-terminal
                # state, so source the final-summary/label side effects
                # directly from the state machine's terminal_actions helper.
                actions = terminal_actions(state, ctx.config)
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
                return 0

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
            # transition() no-ops on a terminal state; emit the terminal
            # side effects (final summary + error label) directly so the
            # operator sees the bailout on the PR, not just in hidden state.
            actions = terminal_actions(state, ctx.config)
            state = self._execute_actions(actions, pr_number, gh_pr, state)
            self._save_state(pr_number, state)
            return 0

        for _ in range(_MAX_TRANSITIONS_PER_RUN):
            if state.status in TERMINAL_STATUSES:
                # State reaching this check was already persisted: either it
                # was just loaded from the state comment (no mutation since),
                # or an earlier iteration saved it before ``continue``. A
                # second save here is a redundant write to GitHub. Check this
                # *before* refetching the PR so the terminal/last iteration
                # doesn't burn a get_pr call.
                return 0

            # Refetch the PR at the top of every iteration: an earlier
            # transition in this same run may have mutated remote state
            # (pushed a new commit, added/removed a label) and downstream
            # snapshot consumers (label-removed safety check, head_sha
            # comparisons) must see the up-to-date view, not the snapshot
            # we captured before the loop started.
            gh_pr = ctx.github.get_pr(pr_number)

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
                self._log_transition(state, next_state)
                state = self._execute_actions(actions, pr_number, gh_pr, next_state)
                self._save_state(pr_number, state)
                continue

            snapshot = self._build_snapshot(state, gh_pr, event=event)
            next_state, actions = transition(state, snapshot, ctx.config, ctx.clock())
            # Clear the cached coder result once the transition has consumed it
            # so a subsequent transition in the same run doesn't see a stale one.
            self._pending_coder_result = None
            self._log_transition(state, next_state)
            state = self._execute_actions(actions, pr_number, gh_pr, next_state)
            self._save_state(pr_number, state)

        # The loop failed to converge within the hard cap. Don't just return:
        # leaving the persisted state in a non-terminal status means every
        # subsequent webhook event re-enters this loop and re-exhausts it,
        # burning CI minutes indefinitely. Drive the state to a terminal
        # ``error`` (emitting the final-summary/label side effects) and save it
        # so future events short-circuit on the terminal check at the top.
        logger.error("Runner exceeded max transitions for PR #%s", pr_number)
        now = ctx.clock()
        state = replace(
            state,
            status="error",
            last_error="max_transitions_exceeded",
            updated_at=now,
        )
        actions = terminal_actions(state, ctx.config)
        state = self._execute_actions(actions, pr_number, gh_pr, state)
        self._save_state(pr_number, state)
        return 1

    # ---- Dry-run planning ----

    def _plan_dry_run(
        self,
        pr_number: int,
        gh_pr: gh_models.PullRequest,
        state: RuntimeState,
        event: ParsedEvent | None,
    ) -> int:
        """Plan one transition and print the actions that would fire.

        Reads (PR, threads, checks) happen normally via ``_build_snapshot``;
        nothing is committed, pushed, posted, edited, labeled, or resolved.
        Always returns 0 — a dry-run is a successful inspection regardless of
        what state the PR is in.
        """
        ctx = self._ctx
        snapshot = self._build_snapshot(state, gh_pr, event=event)
        next_state, actions = transition(state, snapshot, ctx.config, ctx.clock())
        self._log_transition(state, next_state)
        print(
            f"DRY-RUN PR #{pr_number}: status {state.status!r} -> {next_state.status!r}; "
            f"{len(actions)} action(s) planned"
        )
        for action in actions:
            log_action(logger, pr=pr_number, action_type=action.type)
            print(f"  - {_describe_action(action)}")
        return 0

    def _log_transition(self, previous: RuntimeState, nxt: RuntimeState) -> None:
        """Emit a structured ``state_transition`` log when the status changes."""
        if previous.status != nxt.status:
            log_state_transition(
                logger,
                pr=nxt.pr_number,
                from_status=previous.status,
                to_status=nxt.status,
                head_sha=nxt.head_sha,
            )

    # ---- State loading / saving ----

    def _load_or_init_state(self, pr_number: int, gh_pr: gh_models.PullRequest) -> RuntimeState:
        comments = self._ctx.github.get_pr_comments(pr_number)
        sc = find_state_comment([{"id": c.id, "body": c.body} for c in comments])
        if sc is not None:
            self._state_comment_id = int(sc.comment_id) if isinstance(sc.comment_id, int) else None
            self._state_expected_updated_at = sc.state.updated_at
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
        if self._ctx.dry_run:
            # Posting the initial state comment is a mutation; in dry-run we
            # return the in-memory init state without persisting it.
            self._state_expected_updated_at = state.updated_at
            return state
        posted = self._ctx.github.post_comment(pr_number, serialize_state_comment(state))
        self._state_comment_id = posted.id
        self._state_expected_updated_at = state.updated_at
        return state

    def _save_state(self, pr_number: int, state: RuntimeState) -> None:
        # Dry-run performs zero mutations: persisting state would post or edit a
        # PR comment. The single-pass planner never calls this, but guard here
        # too so any future dry-run code path stays mutation-free.
        if self._ctx.dry_run:
            return
        # Fresh PRs without a state comment yet: post once, record id+lock.
        if self._state_comment_id is None:
            body = serialize_state_comment(state)
            posted = self._ctx.github.post_comment(pr_number, body)
            self._state_comment_id = posted.id
            self._state_expected_updated_at = state.updated_at
            return

        # Optimistic concurrency: refetch the current comment body, compare
        # its ``updated_at`` against the value we recorded at load time (or
        # at last successful save). If another runner has mutated it in the
        # interim, raise StateConflictError so the outer loop can bail
        # without overwriting.
        existing = self._fetch_current_state_comment(pr_number)
        if existing is None:
            # Comment vanished (deleted out from under us); post a fresh one.
            body = serialize_state_comment(state)
            posted = self._ctx.github.post_comment(pr_number, body)
            self._state_comment_id = posted.id
            self._state_expected_updated_at = state.updated_at
            return

        expected = self._state_expected_updated_at or existing.state.updated_at
        prepared = prepare_state_comment_update(existing, state, expected_updated_at=expected)
        self._ctx.github.edit_comment(self._state_comment_id, prepared.body)

        # Read-then-PATCH leaves a TOCTOU window: another runner could have
        # edited the comment between our _fetch_current_state_comment above and
        # this edit_comment, and GitHub's issue-comment PATCH has no conditional
        # (If-Match/ETag) support, so the write can't be a true compare-and-swap
        # at the API level. Re-read and confirm our write actually landed (the
        # comment now carries the updated_at we just wrote). If it doesn't, a
        # concurrent runner clobbered us in the window — raise StateConflictError
        # so the run bails cleanly instead of proceeding on a state GitHub no
        # longer reflects. Real serialization should come from a GitHub Actions
        # ``concurrency:`` group keyed by PR number (see CLAUDE.md); this check
        # is the best-effort detection of the residual window.
        verify = self._ctx.github.get_comment(self._state_comment_id)
        if verify is not None:
            verify_sc = find_state_comment([{"id": verify.id, "body": verify.body}])
            if verify_sc is not None and lock_timestamp(
                verify_sc.state.updated_at
            ) != lock_timestamp(state.updated_at):
                raise StateConflictError(
                    "RuntimeState comment was overwritten by another process during save: "
                    f"expected updated_at {state.updated_at.isoformat()}, "
                    f"found {verify_sc.state.updated_at.isoformat()}"
                )
        self._state_expected_updated_at = state.updated_at

    def _fetch_current_state_comment(self, pr_number: int) -> StateComment | None:
        # Fast path: we already know the state comment's id from load/init, so
        # fetch just that comment instead of paging every comment on the PR
        # (which is O(comments) per save and called several times per run).
        if self._state_comment_id is not None:
            comment = self._ctx.github.get_comment(self._state_comment_id)
            if comment is None:
                # The comment was deleted out from under us; signal "vanished"
                # so _save_state reposts a fresh one.
                return None
            return find_state_comment([{"id": comment.id, "body": comment.body}])
        comments = self._ctx.github.get_pr_comments(pr_number)
        return find_state_comment([{"id": c.id, "body": c.body} for c in comments])

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
        remote_head_unverified = False
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
                    # Signal the state machine that the None we just produced
                    # is "unknown", not "absent". Without this flag the
                    # handling transition falls back to ``pr.head_sha`` and
                    # interprets "couldn't verify" as "remote matches".
                    remote_head_unverified = True

        if state.status == "ci_wait":
            # Merge the Checks API and the legacy Statuses API: required checks
            # that report via commit statuses (the same path a ``status``
            # webhook resumes from) never appear in get_check_runs, so without
            # this the gate would see an empty check set and sit in ci_wait
            # until timeout despite the status having reported.
            raw_checks = ctx.github.get_check_runs(state.head_sha)
            raw_statuses = ctx.github.get_commit_statuses(state.head_sha)
            checks = [_convert_check_run(cr, state.head_sha) for cr in (*raw_checks, *raw_statuses)]

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
            remote_head_unverified=remote_head_unverified,
        )

    def _collect_findings(self, state: RuntimeState, gh_pr: gh_models.PullRequest) -> list[Finding]:
        ctx = self._ctx
        all_findings: list[Finding] = []
        # Only collect findings from reviewers triggered in the *current* round.
        # Using the whole trigger_history (or falling back to state.created_at
        # for never-triggered reviewers) would pull in comments from prior
        # rounds — or every comment since the PR opened — and feed stale,
        # out-of-scope findings into ``handling``. This mirrors the
        # current-round filter in ``_all_enabled_reviewers_responded``.
        triggers_by_reviewer: dict[str, ReviewerTrigger] = {}
        for trig in state.trigger_history:
            if trig.round_index != state.round_index:
                continue
            existing = triggers_by_reviewer.get(trig.reviewer_name)
            if existing is None or trig.timestamp > existing.timestamp:
                triggers_by_reviewer[trig.reviewer_name] = trig

        for name, reviewer in ctx.reviewers.items():
            cfg = ctx.config.reviewers.get(name)
            if cfg is not None and not cfg.enabled:
                continue
            trig = triggers_by_reviewer.get(name)
            if trig is None:
                # Reviewer was not triggered this round; it has no findings in
                # scope for the current collection pass.
                continue
            try:
                found = reviewer.collect_findings(state.pr_number, gh_pr.head_sha, trig.timestamp)
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

        # Fail fast if any reviewer triggered in the current round is missing
        # from ctx.reviewers (misconfiguration, missing plugin, failed wiring).
        # We can neither collect its findings nor confirm its response, so the
        # loop is guaranteed to time out — polling for the full reviewer_timeout
        # (up to 15 min) first just burns CI minutes. Route straight to a
        # terminal needs_human with a clear reason. (``_all_enabled_reviewers_
        # responded`` still guards the same case defensively for any other
        # caller.)
        current_round_triggers = [
            t for t in state.trigger_history if t.round_index == state.round_index
        ]
        for trig in current_round_triggers:
            if trig.reviewer_name not in ctx.reviewers:
                logger.error(
                    "Reviewer %s triggered in round %d but missing from ctx.reviewers; "
                    "failing fast to needs_human",
                    trig.reviewer_name,
                    state.round_index,
                )
                now = ctx.clock()
                state = replace(
                    state,
                    status="needs_human",
                    last_error=f"missing_reviewer_adapter:{trig.reviewer_name}",
                    updated_at=now,
                )
                actions = terminal_actions(state, ctx.config)
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
                return state

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
            # Drop the client's within-tick request memo so this iteration sees
            # fresh data, while still collapsing the duplicate get_review_threads
            # reads *inside* the tick (collect_findings + has_responded). The
            # client may not implement the optional cache (e.g. some fakes), so
            # call it defensively.
            reset_cache = getattr(ctx.github, "reset_request_cache", None)
            if callable(reset_cache):
                reset_cache()

            # Refetch the PR each iteration: this loop can span the full
            # reviewer/phase timeout (potentially over an hour), during which an
            # operator may remove the orchestrator label, close the PR, or flip
            # it back to draft. Snapshot consumers (the label-removed safety
            # check, head_sha comparisons) must see the live PR, not the stale
            # view captured before the loop started.
            gh_pr = ctx.github.get_pr(pr_number)
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
        """Return True iff every reviewer triggered in the current round has
        posted *something* after its trigger. Used to detect the zero-findings
        completion case so the state machine can transition to ``done``
        instead of timing out.

        Only reviewers with a ``ReviewerTrigger`` recorded for the current
        ``state.round_index`` participate in this check. A reviewer that is
        enabled in config but was *not* triggered this round (e.g. trigger
        budget exhausted, or newly added between rounds) must not block the
        short-circuit by virtue of having no post-trigger response.
        """
        ctx = self._ctx
        current_triggers = [t for t in state.trigger_history if t.round_index == state.round_index]
        if not current_triggers:
            # Nothing was triggered in the current round — no basis for
            # declaring the phase complete by zero-findings.
            return False

        # If multiple triggers per reviewer somehow land in the same round,
        # keep the newest so ``has_responded`` measures responses after the
        # latest invocation.
        latest_by_reviewer: dict[str, ReviewerTrigger] = {}
        for trig in current_triggers:
            existing = latest_by_reviewer.get(trig.reviewer_name)
            if existing is None or trig.timestamp > existing.timestamp:
                latest_by_reviewer[trig.reviewer_name] = trig

        for name, trig in latest_by_reviewer.items():
            reviewer = ctx.reviewers.get(name)
            if reviewer is None:
                # A reviewer was triggered this round but is absent from
                # ctx.reviewers on resume (config changed, production wiring
                # failed, plugin unavailable). We can neither collect its
                # findings nor probe whether it responded, so we must NOT treat
                # it as "responded" — doing so would let the runner reach
                # done/no_findings while silently dropping a real review. Refuse
                # to short-circuit; the poll loop then proceeds to its timeout,
                # which routes to needs_human so an operator investigates.
                logger.error(
                    "Reviewer %s triggered in round %d but missing from ctx.reviewers; "
                    "cannot confirm response",
                    name,
                    state.round_index,
                )
                return False
            has_responded = getattr(reviewer, "has_responded", None)
            if not callable(has_responded):
                # Reviewer adapter doesn't implement the optional probe; we
                # cannot prove a response, so refuse to short-circuit.
                return False
            try:
                if not has_responded(state.pr_number, trig.timestamp):
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
            log_action(logger, pr=pr_number, action_type=action.type)
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
            # Reviewer trigger actions carry a ``reviewer`` name in the payload
            # (set by the state machine's triggering transition). For those,
            # render the body through the adapter's ``build_trigger_comment`` so
            # the reviewer's machine marker plus round/head metadata are
            # included — the raw ``ReviewerConfig.trigger_comment`` omits them.
            # All other post_pr_comment actions (e.g. plain PR replies) keep the
            # raw-body path.
            reviewer_name = payload.get("reviewer")
            if isinstance(reviewer_name, str):
                adapter = ctx.reviewers.get(reviewer_name)
                if adapter is None:
                    # Fail fast: posting the raw trigger body would still wake
                    # the external reviewer, but the very next polling phase
                    # can't collect its findings (no adapter) and would dead-end
                    # in missing_reviewer_adapter. Refuse to trigger a reviewer
                    # we can't follow up on, and let the run crash-handler /
                    # _poll_reviewers surface needs_human consistently.
                    raise ValueError(
                        f"Cannot post trigger for reviewer '{reviewer_name}': "
                        "no adapter registered in ctx.reviewers"
                    )
                body = adapter.build_trigger_comment(state.round_index, gh_pr.head_sha)
            else:
                body = str(payload.get("body", ""))
            ctx.github.post_comment(pr_number, body)
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
            try:
                self._do_push(gh_pr)
            except Exception:
                # A failed initial push is dangerous in the require_green path:
                # commit_changes has already checkpointed (often ci_wait with
                # head_sha = the local commit), but that commit never reached
                # the remote, so no CI event can ever arrive for it and the PR
                # would sit in ci_wait forever. Persist a terminal needs_human
                # with a clear reason instead of letting the top-level handler
                # return 1 over the stale non-terminal checkpoint. (Push
                # recovery at the top of _run will still retry on a later run
                # if the worktree commit is intact; this just stops the silent
                # wedge.)
                logger.exception("push_branch failed for PR #%s", pr_number)
                now = ctx.clock()
                state = replace(
                    state,
                    status="needs_human",
                    last_error="push_failed",
                    updated_at=now,
                )
                actions = terminal_actions(state, ctx.config)
                state = self._execute_actions(actions, pr_number, gh_pr, state)
                self._save_state(pr_number, state)
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
        # Safety net: cap commits within a single Runner.run invocation at the
        # configured ``max_commits_per_run``. The scope is intentionally
        # per-process: a fresh webhook invocation starts a new run and may add
        # further commits for subsequent rounds (bounded separately by
        # ``max_total_iterations``). Cumulative ``state.commits_made`` is not
        # the gate here for that reason.
        if self._commits_this_run >= ctx.config.safety.max_commits_per_run:
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
        # instead of the pre-commit HEAD. Bump ``updated_at`` too: it doubles as
        # the optimistic-concurrency lock token in ``_save_state``. Leaving it
        # unchanged would let a concurrent runner that loaded the pre-commit
        # state save over this commit without tripping a conflict.
        return replace(
            state,
            head_sha=sha,
            commits_made=[*state.commits_made, sha],
            updated_at=ctx.clock(),
        )

    def _do_push(self, gh_pr: gh_models.PullRequest) -> None:
        ctx = self._ctx
        if ctx.git is None:
            return
        # Defense in depth: fork PRs are hard-blocked at the top of _run, but
        # guard here too so no future code path can push a fork's head_ref to
        # origin (which would write a branch into the BASE repo, not the fork).
        if gh_pr.is_fork:
            raise ValueError(
                f"Cannot push: PR #{gh_pr.number} is from a fork; fork remotes are unsupported"
            )
        # Never fall back to the base branch when head_ref is missing: pushing
        # the PR's commits directly onto ``main``/``master`` bypasses PR
        # controls and can corrupt the base branch. Fail loudly instead — the
        # outer Runner.run handler logs and exits non-zero rather than doing
        # something destructive.
        branch = gh_pr.head_ref
        if not branch:
            raise ValueError(f"Cannot push: head_ref is empty for PR #{gh_pr.number}")
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


def _describe_action(action: PlannedAction) -> str:
    """Render a planned action as a human-readable ``would ...`` line for dry-run."""
    kind = action.type
    payload = action.payload
    if kind == "noop":
        reason = payload.get("reason")
        return f"would do nothing (noop: {reason})" if reason else "would do nothing (noop)"
    if kind == "post_pr_comment":
        reviewer = payload.get("reviewer")
        if isinstance(reviewer, str):
            return f"would post a comment triggering reviewer {reviewer!r}"
        return "would post a PR comment"
    if kind == "post_final_summary":
        return "would post the final summary comment"
    if kind == "update_status_comment":
        return "would update the hidden status comment"
    if kind == "reply_to_thread":
        return f"would reply to review thread {payload.get('thread_id')}"
    if kind == "resolve_thread":
        return f"would resolve review thread {payload.get('thread_id')}"
    if kind == "add_label":
        return f"would add label {payload.get('label')!r}"
    if kind == "remove_label":
        return f"would remove label {payload.get('label')!r}"
    if kind == "commit_changes":
        return "would commit working-tree changes"
    if kind == "push_branch":
        return "would push the branch to origin"
    if kind == "rollback_changes":
        return "would roll back working-tree changes"
    if kind == "invoke_coder":
        finding_ids = payload.get("finding_ids") or []
        return f"would invoke the coder on {len(finding_ids)} finding(s)"
    return f"would execute action {kind!r}"


def _build_final_summary(state: RuntimeState, payload: dict[str, Any]) -> str:
    reason = payload.get("reason") or state.done_reason or state.last_error or "completed"
    cost = state.cost
    summary = (
        f"AI PR Orchestrator: status `{state.status}` — reason: `{reason}`. "
        f"Coder invocations: {cost.coder_invocations}, "
        f"reviewer triggers: {cost.reviewer_triggers}, "
        f"tokens: {cost.input_tokens + cost.output_tokens}."
    )
    # Notify the configured handles for needs_human/error outcomes. The state
    # machine resolves the right list (mention_on_needs_human vs
    # mention_on_error) into the action payload; we just render them. Each
    # handle is normalized to a leading "@" so a config of either "foo" or
    # "@foo" produces a valid mention.
    mentions = payload.get("mentions")
    if isinstance(mentions, list) and mentions:
        handles = " ".join(f"@{m.lstrip('@')}" for m in mentions if isinstance(m, str) and m)
        if handles:
            summary = f"{summary}\n\ncc: {handles}"
    return summary


# ---- CLI entry points ----


def run(*, pr_number: int, dry_run: bool, event_path: Path | None = None) -> int:
    """Run the orchestrator for a pull request.

    Builds dependencies from configuration and the environment, then delegates
    to ``Runner.run``. When ``dry_run`` is set the runner plans a single
    transition and prints the actions it would take, performing zero mutations.
    """
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

    # Configure structured JSON logging with secret redaction. The env-var names
    # the operator allow-listed for the coder (plus GH_TOKEN) are resolved to
    # their values so they never appear verbatim in the log stream.
    setup_logging(
        level=os.environ.get("AIPRO_LOG_LEVEL", "INFO"),
        secrets=collect_secret_values(config.main_coder.env),
    )

    ctx = _build_runtime_context(config, dry_run=dry_run)
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


def _build_runtime_context(_config: Config, *, dry_run: bool = False) -> RunnerContext | None:
    # Real construction of GitHub client / coder / reviewers from environment
    # and config is left to a follow-up issue (V1-12). When wired, ``dry_run``
    # must flow into both the GitHubClient (mutations become no-ops) and the
    # RunnerContext (``git=None``, single-pass planning). The Runner's dry-run
    # behavior itself is fully exercised by unit tests via injected fakes.
    del dry_run
    print(NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return None
