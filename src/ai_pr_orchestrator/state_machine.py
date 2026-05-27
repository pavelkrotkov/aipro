"""Pure state transition logic for the PR review loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from ai_pr_orchestrator.config import Config
from ai_pr_orchestrator.decision_application import apply_decisions
from ai_pr_orchestrator.models import (
    AgentRunResult,
    CheckRun,
    Finding,
    PlannedAction,
    PullRequest,
    ReviewerTrigger,
    RuntimeState,
)

TERMINAL_STATUSES = frozenset({"done", "needs_human", "error"})
PASSING_CHECK_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})


@dataclass(frozen=True)
class TransitionSnapshot:
    """Side-effect-free facts observed outside the state machine."""

    pr: PullRequest
    findings: list[Finding] | None = None
    checks: list[CheckRun] | None = None
    coder_result: AgentRunResult | None = None
    remote_head_sha: str | None = None
    event_head_sha: str | None = None
    reviewer_responded: bool = False
    reviewer_timed_out: bool = False
    worktree_changed: bool = False


def transition(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    """Advance state and return planned side effects without performing I/O.

    Executors must record the new pushed commit SHA in persisted state after a
    successful ``push_branch`` action, before resuming from CI events. The pure
    state machine cannot know that SHA before the side effect runs.
    """
    if state.status in TERMINAL_STATUSES:
        return _touch(state, now), []

    label_action = _label_removed_transition(state, snapshot, config, now)
    if label_action is not None:
        return label_action

    safety_action = _safety_transition(state, snapshot, config, now)
    if safety_action is not None:
        return safety_action

    limit_action = _limit_transition(state, config, now)
    if limit_action is not None:
        return limit_action

    if state.status == "init":
        return (
            _replace_state(
                state,
                now,
                status="triggering",
                round_index=state.round_index + 1,
            ),
            [_status_action("triggering")],
        )
    if state.status == "triggering":
        return _transition_triggering(state, snapshot, config, now)
    if state.status == "waiting":
        return _transition_waiting(state, snapshot, config, now)
    if state.status == "collecting":
        # V1 treats collecting as a persisted alias for the reviewer-response
        # collection branch handled by waiting.
        return _transition_waiting(state, snapshot, config, now)
    if state.status == "handling":
        return _transition_handling(state, snapshot, config, now)
    if state.status == "ci_wait":
        return _transition_ci_wait(state, snapshot, config, now)

    error = _with_status(state, "error", now, last_error=f"unsupported status: {state.status}")
    return error, _terminal_actions(error, config)


def _transition_triggering(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    has_enabled_reviewers = any(reviewer.enabled for reviewer in config.reviewers.values())
    if not has_enabled_reviewers:
        new_state = _with_status(state, "error", now, last_error="no_reviewers_configured")
        return new_state, _terminal_actions(new_state, config)

    enabled_reviewers = [
        (name, reviewer)
        for name, reviewer in config.reviewers.items()
        if reviewer.enabled and reviewer.trigger_comment
    ]
    if (
        state.cost.reviewer_triggers + len(enabled_reviewers)
        > config.safety.max_reviewer_triggers_per_run
    ):
        return _cost_limit_transition(state, config, now)

    actions = [
        PlannedAction(
            "post_pr_comment",
            {
                "body": reviewer.trigger_comment,
                "reviewer": name,
                "pr_number": state.pr_number,
                "head_sha": snapshot.pr.head_sha,
            },
        )
        for name, reviewer in enabled_reviewers
    ]
    triggers = [
        ReviewerTrigger(
            reviewer_name=name,
            round_index=state.round_index,
            timestamp=now,
            head_sha=snapshot.pr.head_sha,
        )
        for name, _reviewer in enabled_reviewers
    ]
    new_cost = replace(
        state.cost,
        reviewer_triggers=state.cost.reviewer_triggers + len(triggers),
        total_api_calls=state.cost.total_api_calls + len(actions),
    )
    new_state = _replace_state(
        state,
        now,
        status="waiting",
        head_sha=snapshot.pr.head_sha,
        base_sha=snapshot.pr.base_sha,
        trigger_history=[*state.trigger_history, *triggers],
        cost=new_cost,
    )
    return new_state, actions


def _transition_waiting(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    findings = snapshot.findings or []
    if findings:
        return _with_status(state, "handling", now), [_status_action("handling")]
    if snapshot.reviewer_timed_out:
        new_state = _with_status(
            state,
            "needs_human",
            now,
            last_error="reviewer_timeout",
        )
        return new_state, _terminal_actions(new_state, config)
    if snapshot.reviewer_responded:
        new_state = _with_status(state, "done", now, done_reason="no_findings")
        return new_state, _terminal_actions(new_state, config)
    return _touch(state, now), [PlannedAction("noop", {"reason": "waiting_for_reviewer"})]


def _transition_handling(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    remote_head_sha = snapshot.remote_head_sha or snapshot.pr.head_sha
    if remote_head_sha != state.head_sha:
        new_state = _replace_state(
            state,
            now,
            status="init",
            head_sha=remote_head_sha,
            base_sha=snapshot.pr.base_sha,
            last_error="stale_coder_output_discarded",
        )
        return new_state, [_status_action("init", reason="head_sha_changed")]

    if snapshot.coder_result is None:
        if state.last_coder_round_index == state.round_index:
            return _touch(state, now), [PlannedAction("noop", {"reason": "waiting_for_coder"})]
        if state.cost.coder_invocations + 1 > config.safety.max_coder_invocations_per_run:
            return _cost_limit_transition(state, config, now)

        new_cost = replace(
            state.cost,
            coder_invocations=state.cost.coder_invocations + 1,
            total_api_calls=state.cost.total_api_calls + 1,
        )
        new_state = _replace_state(
            state,
            now,
            cost=new_cost,
            last_coder_round_index=state.round_index,
        )
        return new_state, [
            PlannedAction(
                "invoke_coder",
                {
                    "pr_number": state.pr_number,
                    "head_sha": state.head_sha,
                    "finding_ids": [finding.id for finding in snapshot.findings or []],
                },
            )
        ]

    result = snapshot.coder_result
    token_usage = result.token_usage
    input_tokens = token_usage.input_tokens if token_usage else 0
    output_tokens = token_usage.output_tokens if token_usage else 0
    new_cost = replace(
        state.cost,
        input_tokens=state.cost.input_tokens + input_tokens,
        output_tokens=state.cost.output_tokens + output_tokens,
    )
    if result.needs_human:
        new_state = _replace_state(
            state,
            now,
            status="needs_human",
            cost=new_cost,
            last_error="coder_needs_human",
        )
        return new_state, _terminal_actions(new_state, config)

    if _has_test_regression(result):
        new_state = _replace_state(
            state,
            now,
            status="needs_human",
            cost=new_cost,
            last_error="test_regression",
        )
        return new_state, [
            PlannedAction("rollback_changes", {"reason": "test_regression"}),
            *_decision_reply_actions(result),
            *_terminal_actions(new_state, config),
        ]

    if result.changed and not snapshot.worktree_changed:
        new_state = _replace_state(
            state,
            now,
            status="error",
            cost=new_cost,
            last_error="coder_reported_changes_but_worktree_clean",
        )
        return new_state, _terminal_actions(new_state, config)

    bot_ids = _bot_finding_ids(snapshot.findings, config)
    app_result = apply_decisions(
        _decisions(result),
        config.thread_policy,
        bot_ids,
        state.handled_findings,
        now,
    )
    handled_findings = app_result.handled_findings
    decision_actions = app_result.actions

    if app_result.has_needs_human:
        new_state = _replace_state(
            state,
            now,
            status="needs_human",
            handled_findings=handled_findings,
            cost=new_cost,
            last_error="decision_needs_human",
        )
        write_actions: list[PlannedAction] = []
        if result.changed:
            write_actions = [
                PlannedAction(
                    "commit_changes",
                    {"message": result.commit_message or config.git.commit_message_prefix},
                ),
                PlannedAction("push_branch", {"head_sha": state.head_sha}),
            ]
        return new_state, [
            *decision_actions,
            *write_actions,
            *_terminal_actions(new_state, config),
        ]

    if result.changed:
        write_actions = [
            PlannedAction(
                "commit_changes",
                {"message": result.commit_message or config.git.commit_message_prefix},
            ),
            PlannedAction("push_branch", {"head_sha": state.head_sha}),
        ]
        if config.ci.require_green_before_done:
            new_state = _replace_state(
                state,
                now,
                status="ci_wait",
                handled_findings=handled_findings,
                cost=new_cost,
                ci_wait_started_at=now,
            )
            return new_state, [*decision_actions, *write_actions, _status_action("ci_wait")]
        new_state = _replace_state(
            state,
            now,
            status="done",
            handled_findings=handled_findings,
            cost=new_cost,
            done_reason="completed",
        )
        return new_state, [*decision_actions, *write_actions, *_terminal_actions(new_state, config)]

    new_state = _replace_state(
        state,
        now,
        status="done",
        handled_findings=handled_findings,
        cost=new_cost,
        done_reason="completed",
    )
    return new_state, [*decision_actions, *_terminal_actions(new_state, config)]


def _transition_ci_wait(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    if snapshot.event_head_sha is not None and snapshot.event_head_sha != state.head_sha:
        return _touch(state, now), [PlannedAction("noop", {"reason": "stale_ci_event"})]

    ci_wait_started_at = state.ci_wait_started_at or state.updated_at
    if now - ci_wait_started_at >= timedelta(seconds=config.ci.timeout_seconds):
        new_state = _with_status(state, "needs_human", now, last_error="ci_timeout")
        return new_state, _terminal_actions(new_state, config)

    checks = _relevant_checks(snapshot.checks or [], config, state.head_sha)
    if not checks or any(
        check.status != "completed" or check.conclusion is None for check in checks
    ):
        return _touch(state, now), [PlannedAction("noop", {"reason": "ci_pending"})]

    if any(check.conclusion not in PASSING_CHECK_CONCLUSIONS for check in checks):
        new_state = _with_status(state, "needs_human", now, last_error="ci_failed")
        return new_state, _terminal_actions(new_state, config)

    new_state = _with_status(state, "done", now, done_reason="ci_passed")
    return new_state, _terminal_actions(new_state, config)


def _label_removed_transition(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]] | None:
    if not config.safety.only_run_on_labeled_prs or config.enabled_label in snapshot.pr.labels:
        return None
    new_state = _with_status(state, "done", now, done_reason="label_removed")
    return new_state, _terminal_actions(new_state, config)


def _safety_transition(
    state: RuntimeState,
    snapshot: TransitionSnapshot,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]] | None:
    if config.safety.disallow_forks and snapshot.pr.is_fork:
        return _error(state, config, now, "fork_pr_not_allowed")
    if snapshot.pr.author_association not in config.safety.allowed_pr_author_associations:
        return _error(
            state,
            config,
            now,
            f"untrusted_author_association:{snapshot.pr.author_association}",
        )
    if config.safety.disallow_workflow_file_changes and any(
        path == ".github/workflows" or path.startswith(".github/workflows/")
        for path in snapshot.pr.changed_files or []
    ):
        new_state = _with_status(state, "needs_human", now, last_error="workflow_file_changed")
        return new_state, _terminal_actions(new_state, config)
    return None


def _limit_transition(
    state: RuntimeState,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]] | None:
    if state.round_index >= config.safety.max_total_iterations:
        new_state = _with_status(state, "needs_human", now, last_error="max_iterations_reached")
        return new_state, _terminal_actions(new_state, config)
    if state.status != "ci_wait" and state.cost.exceeds_limits(config):
        return _cost_limit_transition(state, config, now)
    return None


def _cost_limit_transition(
    state: RuntimeState,
    config: Config,
    now: datetime,
) -> tuple[RuntimeState, list[PlannedAction]]:
    new_state = _with_status(state, "needs_human", now, last_error="cost_limit_reached")
    actions = _terminal_actions(new_state, config)
    return new_state, [
        (
            PlannedAction(
                "post_final_summary",
                {"reason": "cost_limit_reached", "cost": state.cost.to_dict()},
            )
            if action.type == "post_final_summary"
            else action
        )
        for action in actions
    ]


def _error(
    state: RuntimeState,
    config: Config,
    now: datetime,
    reason: str,
) -> tuple[RuntimeState, list[PlannedAction]]:
    new_state = _with_status(state, "error", now, last_error=reason)
    return new_state, _terminal_actions(new_state, config)


def _bot_finding_ids(findings: list[Finding] | None, config: Config) -> frozenset[str]:
    if not findings:
        return frozenset()
    return frozenset(
        f.id
        for f in findings
        if config.reviewers.get(f.source) and config.reviewers[f.source].bot_logins
    )


def _decision_reply_actions(result: AgentRunResult) -> list[PlannedAction]:
    actions: list[PlannedAction] = []
    for decision in _decisions(result):
        if decision.reply and decision.thread_id:
            actions.append(
                PlannedAction(
                    "reply_to_thread",
                    {
                        "finding_id": decision.finding_id,
                        "thread_id": decision.thread_id,
                        "body": decision.reply,
                    },
                )
            )
    return actions


def _terminal_actions(state: RuntimeState, config: Config) -> list[PlannedAction]:
    if state.status == "done":
        return [
            PlannedAction("post_final_summary", {"reason": state.done_reason}),
            PlannedAction("add_label", {"label": config.done_label}),
            PlannedAction("remove_label", {"label": config.enabled_label}),
        ]
    if state.status == "needs_human":
        return [
            PlannedAction("post_final_summary", {"reason": state.last_error}),
            PlannedAction("add_label", {"label": config.error_label}),
        ]
    if state.status == "error":
        return [
            PlannedAction("post_final_summary", {"reason": state.last_error}),
            PlannedAction("add_label", {"label": config.error_label}),
        ]
    return []


def _status_action(status: str, **payload: Any) -> PlannedAction:
    return PlannedAction("update_status_comment", {"status": status, **payload})


def _relevant_checks(checks: list[CheckRun], config: Config, head_sha: str) -> list[CheckRun]:
    required = set(config.ci.required_checks)
    ignored = set(config.ci.ignored_checks)
    return [
        check
        for check in checks
        if check.name not in ignored
        and (not required or check.name in required)
        and (not check.head_sha or check.head_sha == head_sha)
    ]


def _has_test_regression(result: AgentRunResult) -> bool:
    return any(test.result == "failed" for test in result.tests or [])


def _decisions(result: AgentRunResult) -> list[Any]:
    return result.decisions or []


def _with_status(
    state: RuntimeState,
    status: str,
    now: datetime,
    **changes: Any,
) -> RuntimeState:
    return _replace_state(state, now, status=status, **changes)


def _touch(state: RuntimeState, now: datetime) -> RuntimeState:
    return _replace_state(state, now)


def _replace_state(state: RuntimeState, now: datetime, **changes: Any) -> RuntimeState:
    return replace(state, updated_at=now, **changes)
