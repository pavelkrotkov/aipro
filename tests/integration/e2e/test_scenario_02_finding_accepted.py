"""E2E scenario 2 (issue #55): valid blocking finding -> developer fix ->
reviewer accepts -> CI -> PR.

STATUS: blocked on a V3 surface gap. See ``tests/integration/e2e/
KNOWN_V3_GAPS.md`` (or the body of PR #73 when it merges) for the
follow-up. Today the test asserts the *expected* outcome and is
marked xfail; once the V3 fix lands, the markers come off.

Why the gap exists:

The foreman loop's ``_run_lane`` builds a ``LaneExecutionContext``
with ``round_id=state.round_id`` and passes it through
``CaoLaneExecutor`` -> ``CaoSessionController.start_session`` ->
the CAO control plane. ``start_session`` adopts an existing session
under the deterministic name ``(run, lane)`` if one exists, but
rejects adoption when the existing session's stored ``round_id``
does not match the requested one
(``CaoAdoptionMismatchError`` in ``v3/cao.py``).

The first coder call writes the session with ``round_id=None`` (the
state's default). The second coder call (a fix round) requests the
same session name with ``round_id="review-1"`` (the state now carries
the first review round's id). Adoption is refused, the foreman
escalates, and the item is marked ``needs_human``.

The unit tests in ``tests/unit/test_v3_foreman.py`` cover the same
adjudication path with ``ScriptedExecutor`` (which has no
attribution check), so the foreman's review logic is well-tested
in isolation. The E2E version is blocked on the V3 surface gap
until the foreman's session-naming strategy and the controller's
adoption rules are reconciled.

Acceptance (per #55 E2E scenarios, #2, applied when the gap closes):
- ``coder_invocations == 2`` (initial + fix).
- ``review_rounds == 2``.
- One PR opened after the second coder call.
- The finding's disposition history persists in the queue's state.
"""

from __future__ import annotations

import pytest

from ai_pr_orchestrator.v3.domain import ReviewerFinding
from tests.integration._fake_cao_server import (
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_STARTED,
)

#: Status sequence the developer's CAO session walks through. The
#: controller's idle-settle rule maps it onto ``completed``.
_TERMINAL = (STATUS_STARTED, STATUS_PROCESSING, STATUS_IDLE, STATUS_IDLE, STATUS_IDLE)


def _script_worker(fake_cao, run_id: str) -> None:
    name = f"cao-aipro-{run_id}-developer"
    fake_cao.set_status_sequence(name, _TERMINAL)
    fake_cao.set_output(name, "scenario-2 worker output")


# Once the V3 surface gap is fixed, the markers below come off.
_XFAIL_REASON = (
    "V3 surface gap: CaoSessionController refuses to adopt the developer "
    "session on the fix round because the foreman's LaneExecutionContext "
    "round_id changes from None to 'review-1' between the two coder "
    "invocations, and the adoption validation compares round_id. "
    "See the PR-4 body for the follow-up sub-issue."
)


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
def test_scenario_2_blocking_finding_triggers_fix_round(fake_cao, foreman_harness):
    """A major finding surfaces in round 1, the foreman runs the coder
    again, round 2 returns no findings, the foreman reaches ``done``."""
    blocking = ReviewerFinding(
        id="f-blocker",
        lane="requirements-reviewer",
        body="missing input validation",
        severity="blocker",
        run_id="ignored-by-hybrid",
        round_id="ignored-by-hybrid",
    )
    loop, _queue, fake = foreman_harness(
        seed_issue_numbers=[1],
        hybrid_findings={1: [blocking]},
    )
    _script_worker(fake_cao, loop.run_id)

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; "
        f"reason={outcome.reason!r}; escalated={outcome.escalated}"
    )
    # Initial coder call + one fix round.
    assert outcome.coder_invocations == 2, (
        f"expected 2 coder invocations (initial + fix), got {outcome.coder_invocations}"
    )
    assert outcome.review_rounds == 2
    # The PR exists, opened after the second review round.
    open_prs = fake.list_open_prs()
    assert len(open_prs) == 1


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
def test_scenario_2_finding_disposition_history_persists(fake_cao, foreman_harness):
    """The queue's authoritative state carries the disposition history
    so a restarted foreman can see what happened in prior rounds."""
    blocking = ReviewerFinding(
        id="f-2",
        lane="requirements-reviewer",
        body="missing audit log",
        severity="major",
        run_id="ignored-by-hybrid",
        round_id="ignored-by-hybrid",
    )
    loop, queue, _fake = foreman_harness(
        seed_issue_numbers=[2],
        hybrid_findings={1: [blocking]},
    )
    _script_worker(fake_cao, loop.run_id)

    loop.run_pass()

    state = queue.load_state("owner/repo#2")
    assert state is not None
    assert state.phase == "done"
    # Two rounds' worth of findings are persisted.
    assert len(state.findings_history) >= 1
    # The first round's finding was dispositioned as ``fix``.
    first_round = state.findings_history[0]
    assert any(f.id.startswith("f-2") and f.disposition is not None for f in first_round)


def test_scenario_2_no_finding_terminates_quickly(fake_cao, foreman_harness):
    """The hybrid's default (no findings) makes scenario 2 collapse to
    the happy path. Sanity check that the hybrid's no-finding case is
    equivalent to scenario 1; this case does NOT hit the V3 gap
    because only one coder invocation is needed."""
    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[3], hybrid_findings={})
    name = f"cao-aipro-{loop.run_id}-developer"
    fake_cao.set_status_sequence(name, _TERMINAL)
    fake_cao.set_output(name, "scenario-2 worker output")

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done"
    assert outcome.coder_invocations == 1
    assert outcome.review_rounds == 1
