"""Decision application: converts coder verdicts into planned GitHub actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ai_pr_orchestrator.models import Decision, HandledFinding, PlannedAction

if TYPE_CHECKING:
    from ai_pr_orchestrator.config import ThreadPolicyConfig


@dataclass(frozen=True)
class DecisionApplicationResult:
    actions: list[PlannedAction]
    handled_findings: dict[str, HandledFinding]
    has_needs_human: bool


def apply_decisions(
    decisions: list[Decision],
    thread_policy: ThreadPolicyConfig,
    bot_finding_ids: frozenset[str],
    already_handled: dict[str, HandledFinding],
    now: datetime,
) -> DecisionApplicationResult:
    actions: list[PlannedAction] = []
    handled = dict(already_handled)
    has_needs_human = False

    for decision in decisions:
        if decision.finding_id in handled:
            continue

        is_bot = decision.finding_id in bot_finding_ids
        reply_action = _reply_action(decision)
        if reply_action:
            actions.append(reply_action)

        if _should_resolve(decision, is_bot, thread_policy, reply_action is not None):
            actions.append(
                PlannedAction(
                    "resolve_thread",
                    {"finding_id": decision.finding_id, "thread_id": decision.thread_id},
                )
            )

        if decision.verdict == "needs_human":
            has_needs_human = True

        handled[decision.finding_id] = HandledFinding(
            finding_id=decision.finding_id,
            verdict=decision.verdict,
            confidence=decision.confidence,
            reason=decision.reason,
            reply=decision.reply,
            should_resolve=decision.should_resolve,
            changed_files=decision.changed_files or [],
            handled_at=now,
        )

    return DecisionApplicationResult(
        actions=actions,
        handled_findings=handled,
        has_needs_human=has_needs_human,
    )


def _reply_action(decision: Decision) -> PlannedAction | None:
    if not decision.reply:
        return None
    if decision.thread_id:
        return PlannedAction(
            "reply_to_thread",
            {
                "finding_id": decision.finding_id,
                "thread_id": decision.thread_id,
                "body": decision.reply,
            },
        )
    return PlannedAction(
        "post_pr_comment",
        {"finding_id": decision.finding_id, "body": decision.reply},
    )


def _should_resolve(
    decision: Decision,
    is_bot: bool,
    policy: ThreadPolicyConfig,
    has_reply: bool,
) -> bool:
    if not decision.should_resolve or not decision.thread_id:
        return False
    if decision.verdict == "needs_human":
        return False
    if not is_bot and policy.never_resolve_human_threads:
        return False
    if policy.require_reply_before_resolve and not has_reply:
        return False
    if decision.verdict == "accepted":
        return policy.auto_resolve_bot_threads
    if decision.verdict == "rejected":
        return policy.resolve_rejected_bot_threads
    return False
