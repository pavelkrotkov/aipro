"""E2E scenario 3 (issue #55): invalid finding -> developer rebuttal ->
independent acceptance.

STATUS: blocked on TWO V3 surface gaps. See
``tests/integration/e2e/KNOWN_V3_GAPS.md`` (or the body of PR #73
when it merges) for the follow-ups.

Gap A (same as scenario 2): multi-coder-invocation attribution
mismatch. The first coder call writes the session with
``round_id=None``; the rebuttal coder call requests the same session
with ``round_id="review-1"`` and the controller refuses adoption.

Gap B (new): there is no built-in path in the foreman for
"developer rebuts a finding -> independent reviewer accepts". The
foreman today judges by severity (blocker/major -> fix, minor ->
defer) and never by validity. A "rebuttal" path would require a
V3 feature: a disposition of ``rebut`` that, when applied, prompts
an independent reviewer to either confirm or reject the rebuttal.
That is a real product change, not a test fudge.

The unit tests in ``tests/unit/test_v3_foreman.py`` cover the
severity-driven adjudication path; a future "rebuttal" V3 feature
will get its own unit tests. The E2E version is blocked on Gap B
and is a useful documentation of the expected behavior, marked xfail.

Acceptance (per #55 E2E scenarios, #3, applied when the gaps close):
- Round 1 produces a finding the foreman judges invalid (severity
  major, but the developer rebuts).
- The foreman runs the coder (rebuttal), then triggers an
  independent reviewer round.
- The independent reviewer accepts the rebuttal, the foreman
  reaches ``done`` without re-running the coder for the same
  finding.
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
    fake_cao.set_output(name, "scenario-3 worker output")


_XFAIL_REASON = (
    "V3 surface gap: CaoSessionController refuses to adopt the developer "
    "session on the rebuttal round (round_id changes from None to "
    "'review-1'), AND the foreman has no built-in 'developer rebuttal' "
    "path today — adjudication is severity-driven only. Both gaps must "
    "close before this scenario can ship. See the PR-4 body."
)


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
def test_scenario_3_invalid_finding_triggers_rebuttal(fake_cao, foreman_harness):
    """Round 1 surfaces a major finding the developer rebuts; round 2's
    independent reviewer accepts the rebuttal; the foreman reaches
    ``done`` without running the coder a second time for the same
    finding."""
    suspect = ReviewerFinding(
        id="f-suspect",
        lane="requirements-reviewer",
        body="alleged missing validation",
        severity="major",
        run_id="ignored-by-hybrid",
        round_id="ignored-by-hybrid",
    )
    # Round 1: suspect finding surfaces. Round 2: independent reviewer
    # accepts the rebuttal (no findings -> foreman reaches done).
    loop, queue, _fake = foreman_harness(
        seed_issue_numbers=[1],
        hybrid_findings={1: [suspect], 2: []},
    )
    _script_worker(fake_cao, loop.run_id)

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; "
        f"reason={outcome.reason!r}; escalated={outcome.escalated}"
    )
    # The rebuttal path requires the foreman to run the coder once
    # more, then the independent reviewer round closes it.
    assert outcome.coder_invocations >= 1
    assert outcome.review_rounds >= 2
    # The suspect finding's disposition should be ``rebut`` (or
    # equivalent), not ``fix``.
    state = queue.load_state("owner/repo#1")
    assert state is not None
    assert any(d.action == "rebut" for d in state.dispositions), (
        f"expected a 'rebut' disposition, got {[d.action for d in state.dispositions]}"
    )
