from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.config import ThreadPolicyConfig
from ai_pr_orchestrator.decision_application import DecisionApplicationResult, apply_decisions
from ai_pr_orchestrator.models import Decision, HandledFinding

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)
BOT_IDS: frozenset[str] = frozenset({"f1"})
NO_BOTS: frozenset[str] = frozenset()


def _policy(**overrides: Any) -> ThreadPolicyConfig:
    defaults: dict[str, Any] = {
        "auto_resolve_bot_threads": True,
        "never_resolve_human_threads": True,
        "resolve_rejected_bot_threads": True,
        "require_reply_before_resolve": True,
    }
    defaults.update(overrides)
    return ThreadPolicyConfig(**defaults)  # type: ignore[arg-type]


def _decision(**overrides: Any) -> Decision:
    defaults: dict[str, Any] = {
        "finding_id": "f1",
        "verdict": "accepted",
        "confidence": "high",
        "reason": "valid issue",
        "reply": "Fixed in abc123",
        "should_resolve": True,
        "thread_id": "thread-1",
        "changed_files": ["src/app.py"],
    }
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def _action_types(result: DecisionApplicationResult) -> list[str]:
    return [a.type for a in result.actions]


# --- Accepted findings ---


def test_accepted_plans_reply_to_thread() -> None:
    result = apply_decisions([_decision()], _policy(), BOT_IDS, {}, NOW)
    assert "reply_to_thread" in _action_types(result)
    reply = next(a for a in result.actions if a.type == "reply_to_thread")
    assert reply.payload["body"] == "Fixed in abc123"


def test_accepted_plans_resolve_for_bot_thread() -> None:
    result = apply_decisions([_decision()], _policy(), BOT_IDS, {}, NOW)
    assert "resolve_thread" in _action_types(result)
    resolve = next(a for a in result.actions if a.type == "resolve_thread")
    assert resolve.payload["thread_id"] == "thread-1"


def test_accepted_records_handled_finding() -> None:
    result = apply_decisions([_decision()], _policy(), BOT_IDS, {}, NOW)
    assert "f1" in result.handled_findings
    hf = result.handled_findings["f1"]
    assert hf.verdict == "accepted"
    assert hf.handled_at == NOW


# --- Rejected findings ---


def test_rejected_plans_reply_with_rebuttal() -> None:
    d = _decision(verdict="rejected", reply="Not a real issue because ...")
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert "reply_to_thread" in _action_types(result)
    reply = next(a for a in result.actions if a.type == "reply_to_thread")
    assert reply.payload["body"] == "Not a real issue because ..."


def test_rejected_resolves_bot_thread_when_policy_allows() -> None:
    d = _decision(verdict="rejected")
    result = apply_decisions([d], _policy(resolve_rejected_bot_threads=True), BOT_IDS, {}, NOW)
    assert "resolve_thread" in _action_types(result)


def test_rejected_does_not_resolve_when_policy_disallows() -> None:
    d = _decision(verdict="rejected")
    result = apply_decisions([d], _policy(resolve_rejected_bot_threads=False), BOT_IDS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)


def test_rejected_records_handled_finding() -> None:
    d = _decision(verdict="rejected")
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert result.handled_findings["f1"].verdict == "rejected"


# --- Needs-human findings ---


def test_needs_human_plans_reply() -> None:
    d = _decision(verdict="needs_human", reply="Requires architectural decision")
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert "reply_to_thread" in _action_types(result)
    reply = next(a for a in result.actions if a.type == "reply_to_thread")
    assert reply.payload["body"] == "Requires architectural decision"


def test_needs_human_does_not_resolve() -> None:
    d = _decision(verdict="needs_human")
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)


def test_needs_human_propagates_flag() -> None:
    d = _decision(verdict="needs_human")
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert result.has_needs_human is True


# --- Thread policy enforcement ---


def test_never_resolves_human_authored_threads() -> None:
    d = _decision(verdict="accepted")
    result = apply_decisions([d], _policy(), NO_BOTS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)
    assert "reply_to_thread" in _action_types(result)


def test_require_reply_before_resolve_blocks_resolve_when_no_reply() -> None:
    d = _decision(reply="", verdict="accepted")
    result = apply_decisions([d], _policy(require_reply_before_resolve=True), BOT_IDS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)


def test_resolve_allowed_without_reply_when_policy_disabled() -> None:
    d = _decision(reply="", verdict="accepted")
    result = apply_decisions([d], _policy(require_reply_before_resolve=False), BOT_IDS, {}, NOW)
    assert "resolve_thread" in _action_types(result)


def test_reply_comes_before_resolve_in_action_list() -> None:
    result = apply_decisions([_decision()], _policy(), BOT_IDS, {}, NOW)
    types = _action_types(result)
    reply_idx = types.index("reply_to_thread")
    resolve_idx = types.index("resolve_thread")
    assert reply_idx < resolve_idx


# --- Edge cases ---


def test_finding_without_thread_id_plans_post_pr_comment() -> None:
    d = _decision(thread_id=None)
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert "post_pr_comment" in _action_types(result)
    assert "reply_to_thread" not in _action_types(result)
    assert "resolve_thread" not in _action_types(result)


def test_already_handled_finding_is_skipped() -> None:
    existing = {
        "f1": HandledFinding(
            finding_id="f1",
            verdict="accepted",
            confidence="high",
            reason="done",
            reply="Fixed",
            should_resolve=True,
            handled_at=NOW,
        )
    }
    result = apply_decisions([_decision()], _policy(), BOT_IDS, existing, NOW)
    assert result.actions == []
    assert result.handled_findings == existing


def test_multiple_decisions_applied_in_deterministic_order() -> None:
    decisions = [
        _decision(finding_id="f1", reply="Fix 1"),
        _decision(finding_id="f2", reply="Fix 2", thread_id="thread-2"),
    ]
    bot_ids = frozenset({"f1", "f2"})
    result = apply_decisions(decisions, _policy(), bot_ids, {}, NOW)

    types = _action_types(result)
    assert types == [
        "reply_to_thread",
        "resolve_thread",
        "reply_to_thread",
        "resolve_thread",
    ]
    assert result.actions[0].payload["finding_id"] == "f1"
    assert result.actions[2].payload["finding_id"] == "f2"
    assert "f1" in result.handled_findings
    assert "f2" in result.handled_findings


def test_accepted_without_should_resolve_does_not_resolve() -> None:
    d = _decision(should_resolve=False)
    result = apply_decisions([d], _policy(), BOT_IDS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)
    assert "reply_to_thread" in _action_types(result)


def test_auto_resolve_bot_threads_false_blocks_accepted_resolve() -> None:
    d = _decision(verdict="accepted")
    result = apply_decisions([d], _policy(auto_resolve_bot_threads=False), BOT_IDS, {}, NOW)
    assert "resolve_thread" not in _action_types(result)
