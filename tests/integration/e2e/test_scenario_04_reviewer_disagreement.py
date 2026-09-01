"""E2E scenario 4 (issue #55): reviewer disagreement -> stronger /
different-family adjudication.

STATUS: blocked on the same V3 surface gap as scenarios 2 and 3
(multi-coder-invocation attribution mismatch). The unit tests in
``tests/unit/test_v3_foreman.py`` already cover the conflict-group
adjudication path (severity-driven ``fix`` for any conflict
member); the E2E version is marked xfail until the foreman's
session-naming and the controller's adoption rules are reconciled.

Acceptance (per #55 E2E scenarios, #4, applied when the gap closes):
- Round 1 produces conflicting findings from two reviewer lanes on
  the same logical issue.
- The foreman adjudicates the whole conflict group as ``fix`` (no
  averaging) and runs the coder.
- Round 2 returns no findings; the foreman reaches ``done``.
"""

from __future__ import annotations

import pytest

from ai_pr_orchestrator.v3.domain import ReviewerFinding
from tests.integration._fake_cao_server import (
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_STARTED,
)

_TERMINAL = (STATUS_STARTED, STATUS_PROCESSING, STATUS_IDLE, STATUS_IDLE, STATUS_IDLE)


def _script_worker(fake_cao, run_id: str) -> None:
    name = f"cao-aipro-{run_id}-developer"
    fake_cao.set_status_sequence(name, _TERMINAL)
    fake_cao.set_output(name, "scenario-4 worker output")


_XFAIL_REASON = (
    "V3 surface gap: CaoSessionController refuses to adopt the developer "
    "session on the fix round (round_id changes from None to "
    "'review-1'). The conflict-group adjudication itself works in unit "
    "tests; only the E2E round-trip through the real CaoLaneExecutor "
    "is blocked."
)


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
def test_scenario_4_reviewer_disagreement_picks_stronger(fake_cao, foreman_harness):
    """Two reviewer lanes report conflicting findings on the same
    logical issue. The foreman adjudicates the whole conflict group
    as ``fix`` (no averaging) and reaches ``done`` after a fix round."""
    # PR #73 review thread 3 / issue #73: the two findings must share a
    # ``path`` and overlapping line ranges with incompatible claims so
    # ``FindingRegistry.detect_conflicts`` (which clears the incoming
    # ``conflict_group_id`` and recomputes by path + line range) can
    # actually group them. The previous test preassigned group IDs that
    # the production code never reads.
    conflicting_a = ReviewerFinding(
        id="f-conflict-a",
        lane="requirements-reviewer",
        body="needs eager validation",
        severity="major",
        run_id="ignored-by-hybrid",
        round_id="ignored-by-hybrid",
        path="src/validation.py",
        line=10,
        line_end=20,
        claim="validation must run eagerly on entry",
    )
    conflicting_b = ReviewerFinding(
        id="f-conflict-b",
        lane="breaker-reviewer",
        body="needs lazy validation",
        severity="major",
        run_id="ignored-by-hybrid",
        round_id="ignored-by-hybrid",
        path="src/validation.py",
        line=12,
        line_end=18,
        claim="validation must run lazily on demand",
    )

    loop, queue, _fake = foreman_harness(
        seed_issue_numbers=[1],
        hybrid_findings={1: [conflicting_a, conflicting_b]},
    )
    _script_worker(fake_cao, loop.run_id)

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; "
        f"reason={outcome.reason!r}; escalated={outcome.escalated}"
    )
    # The fix round runs the coder once, the second round returns
    # no findings and closes.
    assert outcome.coder_invocations == 2
    assert outcome.review_rounds == 2
    # Both conflict members are dispositioned as ``fix`` (no
    # averaging): the foreman's adjudication is whole-group.
    state = queue.load_state("owner/repo#1")
    assert state is not None
    actions = [d.action for d in state.dispositions]
    assert actions.count("fix") >= 2, (
        f"expected at least 2 'fix' dispositions (one per conflict member), got {actions}"
    )
