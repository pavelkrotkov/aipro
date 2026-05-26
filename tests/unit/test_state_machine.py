from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.config import (
    CiConfig,
    Config,
    MainCoderConfig,
    ReviewerConfig,
    SafetyConfig,
)
from ai_pr_orchestrator.models import (
    AgentRunResult,
    CheckRun,
    CostTracker,
    Decision,
    Finding,
    PullRequest,
    RuntimeState,
    TestResult,
    TokenUsage,
)
from ai_pr_orchestrator.state_machine import TransitionSnapshot, transition

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


def make_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "main_coder": MainCoderConfig(provider="codex_cli"),
        "reviewers": {
            "gemini": ReviewerConfig(
                bot_logins=["gemini-code-assist[bot]"],
                trigger_comment="/gemini review",
            )
        },
    }
    defaults.update(overrides)
    return Config(**defaults)


def make_state(**overrides: Any) -> RuntimeState:
    defaults: dict[str, Any] = {
        "version": 1,
        "pr_number": 4,
        "head_sha": "head-1",
        "base_sha": "base-1",
        "status": "init",
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return RuntimeState(**defaults)


def make_pr(**overrides: Any) -> PullRequest:
    defaults: dict[str, Any] = {
        "number": 4,
        "head_sha": "head-1",
        "base_sha": "base-1",
        "title": "Test PR",
        "author_login": "pavel",
        "author_association": "OWNER",
        "labels": ["ai-loop"],
        "changed_files": ["src/app.py"],
    }
    defaults.update(overrides)
    return PullRequest(**defaults)


def make_snapshot(**overrides: Any) -> TransitionSnapshot:
    defaults: dict[str, Any] = {"pr": make_pr()}
    defaults.update(overrides)
    return TransitionSnapshot(**defaults)


def make_finding(id_: str = "f1") -> Finding:
    return Finding(id=id_, source="gemini", body="Fix this", created_at=NOW, head_sha="head-1")


def make_result(**overrides: Any) -> AgentRunResult:
    defaults: dict[str, Any] = {
        "changed": True,
        "summary": "fixed",
        "decisions": [
            Decision(
                finding_id="f1",
                verdict="accepted",
                confidence="high",
                reason="valid",
                reply="Fixed",
                should_resolve=True,
                thread_id="thread-1",
                changed_files=["src/app.py"],
            )
        ],
        "commit_message": "fix: address review",
    }
    defaults.update(overrides)
    return AgentRunResult(**defaults)


def action_types(actions: list[Any]) -> list[str]:
    return [action.type for action in actions]


def test_init_moves_to_triggering_when_label_present_and_safety_passes() -> None:
    state, actions = transition(make_state(), make_snapshot(), make_config(), NOW)

    assert state.status == "triggering"
    assert state.round_index == 1
    assert action_types(actions) == ["update_status_comment"]


def test_triggering_records_reviewer_trigger_and_waits() -> None:
    state, actions = transition(
        make_state(status="triggering"), make_snapshot(), make_config(), NOW
    )

    assert state.status == "waiting"
    assert state.cost.reviewer_triggers == 1
    assert state.trigger_history[0].reviewer_name == "gemini"
    assert action_types(actions) == ["post_pr_comment"]


def test_triggering_errors_when_no_reviewers_are_configured() -> None:
    state, actions = transition(
        make_state(status="triggering"),
        make_snapshot(),
        make_config(reviewers={}),
        NOW,
    )

    assert state.status == "error"
    assert state.last_error == "no_reviewers_configured"
    assert action_types(actions) == ["post_final_summary", "add_label"]


def test_waiting_moves_to_handling_when_findings_are_collected() -> None:
    state, actions = transition(
        make_state(status="waiting"),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "handling"
    assert action_types(actions) == ["update_status_comment"]


def test_collecting_uses_reviewer_collection_logic() -> None:
    state, actions = transition(
        make_state(status="collecting"),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "handling"
    assert action_types(actions) == ["update_status_comment"]


def test_waiting_finishes_when_reviewer_responded_without_findings() -> None:
    state, actions = transition(
        make_state(status="waiting"),
        make_snapshot(reviewer_responded=True, findings=[]),
        make_config(),
        NOW,
    )

    assert state.status == "done"
    assert state.done_reason == "no_findings"
    assert action_types(actions) == ["post_final_summary", "add_label", "remove_label"]


def test_handling_plans_coder_invocation_before_result_exists() -> None:
    state, actions = transition(
        make_state(status="handling", round_index=1),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "handling"
    assert state.cost.coder_invocations == 1
    assert state.last_coder_round_index == 1
    assert action_types(actions) == ["invoke_coder"]


def test_handling_waits_when_coder_was_already_invoked_for_round() -> None:
    state, actions = transition(
        make_state(
            status="handling",
            round_index=1,
            cost=CostTracker(coder_invocations=1),
            last_coder_round_index=1,
        ),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "handling"
    assert state.cost.coder_invocations == 1
    assert action_types(actions) == ["noop"]
    assert actions[0].payload["reason"] == "waiting_for_coder"


def test_handling_uses_last_coder_round_not_cumulative_invocation_count() -> None:
    state, actions = transition(
        make_state(
            status="handling",
            round_index=2,
            cost=CostTracker(coder_invocations=1),
            last_coder_round_index=2,
        ),
        make_snapshot(findings=[make_finding()]),
        make_config(safety=SafetyConfig(max_coder_invocations_per_run=3)),
        NOW,
    )

    assert state.status == "handling"
    assert state.cost.coder_invocations == 1
    assert action_types(actions) == ["noop"]
    assert actions[0].payload["reason"] == "waiting_for_coder"


def test_handling_commits_pushes_and_waits_for_ci_when_gate_enabled() -> None:
    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(
            findings=[make_finding()],
            coder_result=make_result(),
            worktree_changed=True,
        ),
        make_config(),
        NOW,
    )

    assert state.status == "ci_wait"
    assert state.handled_findings["f1"].verdict == "accepted"
    assert action_types(actions) == [
        "reply_to_thread",
        "resolve_thread",
        "commit_changes",
        "push_branch",
        "update_status_comment",
    ]


def test_handling_uses_configured_commit_message_when_result_has_no_message() -> None:
    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(
            coder_result=make_result(commit_message=None),
            worktree_changed=True,
        ),
        make_config(),
        NOW,
    )

    assert state.status == "ci_wait"
    commit_action = next(action for action in actions if action.type == "commit_changes")
    assert commit_action.payload["message"] == "fix: address AI review feedback"


def test_handling_skips_thread_actions_when_decision_has_no_thread_id() -> None:
    decision = Decision(
        finding_id="f1",
        verdict="accepted",
        confidence="high",
        reason="valid",
        reply="Fixed",
        should_resolve=True,
    )

    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(coder_result=make_result(changed=False, decisions=[decision])),
        make_config(),
        NOW,
    )

    assert state.status == "done"
    assert state.handled_findings["f1"].verdict == "accepted"
    assert "reply_to_thread" not in action_types(actions)
    assert "resolve_thread" not in action_types(actions)


def test_handling_finishes_when_ci_gate_disabled() -> None:
    config = make_config(ci=CiConfig(require_green_before_done=False))

    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(coder_result=make_result(), worktree_changed=True),
        config,
        NOW,
    )

    assert state.status == "done"
    assert action_types(actions)[-3:] == ["post_final_summary", "add_label", "remove_label"]


def test_ci_wait_finishes_when_ci_passed() -> None:
    state, actions = transition(
        make_state(status="ci_wait"),
        make_snapshot(
            checks=[
                CheckRun(
                    id="1",
                    name="test",
                    status="completed",
                    conclusion="success",
                    head_sha="head-1",
                )
            ]
        ),
        make_config(),
        NOW,
    )

    assert state.status == "done"
    assert state.done_reason == "ci_passed"
    assert action_types(actions) == ["post_final_summary", "add_label", "remove_label"]


def test_ci_wait_needs_human_when_ci_failed() -> None:
    state, actions = transition(
        make_state(status="ci_wait"),
        make_snapshot(
            checks=[
                CheckRun(
                    id="1",
                    name="test",
                    status="completed",
                    conclusion="failure",
                    head_sha="head-1",
                )
            ]
        ),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "ci_failed"
    assert action_types(actions) == ["post_final_summary", "add_label"]


def test_label_removal_at_any_non_terminal_state_finishes() -> None:
    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(pr=make_pr(labels=[]), coder_result=make_result()),
        make_config(),
        NOW,
    )

    assert state.status == "done"
    assert state.done_reason == "label_removed"
    assert action_types(actions) == ["post_final_summary", "add_label", "remove_label"]


def test_fork_pr_fails_safety_before_state_logic() -> None:
    state, actions = transition(
        make_state(status="waiting"),
        make_snapshot(pr=make_pr(is_fork=True), findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "error"
    assert state.last_error == "fork_pr_not_allowed"
    assert action_types(actions) == ["post_final_summary", "add_label"]


def test_untrusted_author_fails_safety() -> None:
    state, _actions = transition(
        make_state(),
        make_snapshot(pr=make_pr(author_association="NONE")),
        make_config(),
        NOW,
    )

    assert state.status == "error"
    assert state.last_error == "untrusted_author_association:NONE"


def test_workflow_file_change_needs_human() -> None:
    state, _actions = transition(
        make_state(),
        make_snapshot(pr=make_pr(changed_files=[".github/workflows/ci.yml"])),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "workflow_file_changed"


def test_coder_invocation_limit_blocks_handling_before_invoke() -> None:
    state, actions = transition(
        make_state(status="handling", round_index=2, cost=CostTracker(coder_invocations=1)),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "cost_limit_reached"
    assert action_types(actions) == ["post_final_summary", "add_label"]


def test_coder_invocation_at_limit_does_not_block_passive_ci_wait() -> None:
    state, actions = transition(
        make_state(status="ci_wait", cost=CostTracker(coder_invocations=1)),
        make_snapshot(),
        make_config(),
        NOW,
    )

    assert state.status == "ci_wait"
    assert action_types(actions) == ["noop"]
    assert actions[0].payload["reason"] == "ci_pending"


def test_reviewer_trigger_limit_blocks_triggering() -> None:
    state, _actions = transition(
        make_state(status="triggering", cost=CostTracker(reviewer_triggers=3)),
        make_snapshot(),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "cost_limit_reached"


def test_token_limit_blocks_before_coder_invocation() -> None:
    state, actions = transition(
        make_state(status="handling", cost=CostTracker(input_tokens=90_000, output_tokens=10_000)),
        make_snapshot(findings=[make_finding()]),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert action_types(actions) == ["post_final_summary", "add_label"]


def test_round_limit_needs_human() -> None:
    state, _actions = transition(
        make_state(round_index=3),
        make_snapshot(),
        make_config(safety=SafetyConfig(max_total_iterations=3)),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "max_iterations_reached"


def test_head_sha_race_discards_coder_output_and_reinitializes() -> None:
    state, actions = transition(
        make_state(status="handling", head_sha="old-head"),
        make_snapshot(
            pr=make_pr(head_sha="new-head"),
            remote_head_sha="new-head",
            coder_result=make_result(),
            worktree_changed=True,
        ),
        make_config(),
        NOW,
    )

    assert state.status == "init"
    assert state.head_sha == "new-head"
    assert state.last_error == "stale_coder_output_discarded"
    assert action_types(actions) == ["update_status_comment"]


def test_ci_wait_ignores_stale_event_sha() -> None:
    state, actions = transition(
        make_state(status="ci_wait"),
        make_snapshot(
            event_head_sha="old-head",
            checks=[
                CheckRun(
                    id="1",
                    name="test",
                    status="completed",
                    conclusion="success",
                    head_sha="old-head",
                )
            ],
        ),
        make_config(),
        NOW,
    )

    assert state.status == "ci_wait"
    assert actions[0].payload["reason"] == "stale_ci_event"


def test_waiting_timeout_needs_human() -> None:
    state, _actions = transition(
        make_state(status="waiting"),
        make_snapshot(reviewer_timed_out=True),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "reviewer_timeout"


def test_coder_needs_human_is_terminal() -> None:
    state, _actions = transition(
        make_state(status="handling"),
        make_snapshot(coder_result=make_result(needs_human=True), worktree_changed=False),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert state.last_error == "coder_needs_human"


def test_test_regression_plans_rollback() -> None:
    state, actions = transition(
        make_state(status="handling"),
        make_snapshot(
            coder_result=make_result(tests=[TestResult(command="pytest", result="failed")]),
            worktree_changed=True,
        ),
        make_config(),
        NOW,
    )

    assert state.status == "needs_human"
    assert action_types(actions) == ["rollback_changes", "post_final_summary", "add_label"]


def test_changed_result_with_clean_worktree_is_error() -> None:
    state, _actions = transition(
        make_state(status="handling"),
        make_snapshot(coder_result=make_result(changed=True), worktree_changed=False),
        make_config(),
        NOW,
    )

    assert state.status == "error"
    assert state.last_error == "coder_reported_changes_but_worktree_clean"


def test_handling_records_token_usage_from_coder_result() -> None:
    state, _actions = transition(
        make_state(status="handling"),
        make_snapshot(
            coder_result=make_result(
                changed=False,
                token_usage=TokenUsage(input_tokens=11, output_tokens=7),
            )
        ),
        make_config(),
        NOW,
    )

    assert state.cost.input_tokens == 11
    assert state.cost.output_tokens == 7


def test_already_terminal_states_do_not_replan_actions() -> None:
    state, actions = transition(
        make_state(status="done", done_reason="completed"),
        make_snapshot(),
        make_config(),
        NOW,
    )

    assert state.status == "done"
    assert action_types(actions) == []
