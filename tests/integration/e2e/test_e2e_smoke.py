"""E2E smoke test: the foreman loop drives one fake CAO session to
completion.

Issue #55, P1, PR-3 of 8. The smoke test proves the shared fixtures in
``tests/integration/conftest.py`` wire the foreman loop to the real
``CaoLaneExecutor`` and the in-process ``FakeCAOServer``, and that the
resulting foreman pass walks an issue from claim to ``done``.

This is intentionally minimal: the 12 E2E scenarios in #55 live in
``tests/integration/e2e/test_*.py`` (PRs 4-6) and assert on observable
behaviour — branches, PRs, labels — not on internal foreman attributes.
The smoke test asserts on the foreman outcome (``done``) and on the
labels the queue applied, which is the smallest end-to-end
demonstration possible.
"""

from __future__ import annotations

import pytest

from ai_pr_orchestrator.v3.cao import session_name_for
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE
from tests.integration._fake_cao_server import (
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_STARTED,
)
from tests.integration.conftest import E2E_WORK_TAG

#: A scripted status sequence that walks the controller through
#: ``started -> processing -> idle (x3)``, which the controller's
#: idle-settle rule maps onto the normalized ``completed`` lifecycle.
_DEFAULT_TERMINAL_SEQUENCE = (
    STATUS_STARTED,
    STATUS_PROCESSING,
    STATUS_IDLE,
    STATUS_IDLE,
    STATUS_IDLE,
)


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_foreman_drives_one_fake_cao_session_to_completion(fake_cao, foreman_harness):
    """One seeded issue walks the full lifecycle to ``done`` with the
    real :class:`CaoLaneExecutor` and the in-process ``FakeCAOServer``.

    The CAO session's status sequence is scripted to a clean
    processing-to-idle transition; the controller's idle-settle rule
    reports ``completed`` and the executor's :class:`LaneResult` flows
    through the foreman's review and CI gate.
    """
    loop, queue, fake = foreman_harness(seed_issue_numbers=[1])

    # The foreman's run_id is the source of truth for the deterministic
    # CAO session name; script the fake against the *loop's* run id,
    # not a local one, so the controller's ``session_name_for`` lookup
    # hits the scripted session.
    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    # The foreman reaches the ``done`` phase, the gate passed, and the
    # queue persisted authoritative state.
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; terminal_reason={outcome.terminal_reason!r}"
    )
    assert outcome.gate is not None and outcome.gate.passed
    # The queue advanced the labels.
    state = queue.load_state("owner/repo#1")
    assert state is not None and state.phase == "done"
    assert "v3-work-done" in fake.get_labels(1)
    assert E2E_WORK_TAG not in fake.get_labels(1)


@pytest.mark.usefixtures("cao_lane_executor", "lane_registry")
def test_foreman_records_cao_session_in_fake(fake_cao, foreman_harness):
    """The :class:`CaoLaneExecutor` issued at least one CAO session for
    the developer lane during the foreman pass, with the deterministic
    name the controller derives from the run id and lane."""
    loop, _queue, _fake = foreman_harness(seed_issue_numbers=[1])

    run_id = loop.run_id
    session_name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(session_name, _DEFAULT_TERMINAL_SEQUENCE)
    fake_cao.set_output(session_name, "ok")

    loop.run_pass()

    # The foreman issued at least one CAO session under the developer's
    # deterministic name. We do not assert on the reviewer sessions
    # here because that depends on the foreman's lane choice, which the
    # smoke test does not need to fix; the important contract is that
    # the developer lane spoke to CAO.
    assert session_name in fake_cao._sessions, (
        f"expected CAO session {session_name!r} to exist on the fake "
        f"after the foreman pass; got {list(fake_cao._sessions)}"
    )
