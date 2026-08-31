"""E2E scenario 1 (issue #55): simple issue -> developer -> 3 reviews
-> clean -> CI -> PR.

The foreman loop drives a single seeded issue through the full
lifecycle to ``done``. The worker lane runs through the real
``CaoLaneExecutor`` against ``FakeCAOServer``; the reviewer lanes
return no findings (the hybrid executor's default for round 1).

Acceptance (per #55 E2E scenarios, #1):
- One PR is opened, no duplicates.
- The foreman reaches ``done`` with CI green.
- All labels move through the lifecycle (enabled -> active -> done).
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

#: Status sequence the developer's CAO session walks through. The
#: controller's idle-settle rule maps it onto ``completed``.
_TERMINAL = (STATUS_STARTED, STATUS_PROCESSING, STATUS_IDLE, STATUS_IDLE, STATUS_IDLE)


def _script_worker(fake_cao, run_id: str) -> str:
    """Set up the fake's per-session state for the worker lane and
    return the deterministic session name."""
    name = session_name_for(run_id, DEVELOPER_LANE)
    fake_cao.set_status_sequence(name, _TERMINAL)
    fake_cao.set_output(name, "scenario-1 worker output")
    return name


def test_scenario_1_happy_path_opens_exactly_one_pr(fake_cao, foreman_harness):
    """The foreman walk a single issue from claim to ``done``; exactly
    one open PR is produced and the developer lane spoke to CAO."""
    loop, _queue, fake = foreman_harness(seed_issue_numbers=[1])
    _script_worker(fake_cao, loop.run_id)

    outcomes = loop.run_pass()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.final_phase == "done", (
        f"expected 'done', got {outcome.final_phase!r}; terminal_reason={outcome.terminal_reason!r}"
    )
    assert outcome.gate is not None and outcome.gate.passed

    # Exactly one open PR for issue 1; no duplicates from re-runs.
    open_prs = [pr for pr in fake.list_open_prs() if pr.body and "1" in pr.title]
    assert len(open_prs) == 1, (
        f"expected exactly 1 open PR for issue 1, got {len(open_prs)}: {open_prs}"
    )

    # The worker spoke to CAO under the deterministic session name.
    name = session_name_for(loop.run_id, DEVELOPER_LANE)
    assert name in fake_cao._sessions


def test_scenario_1_advances_labels_through_lifecycle(fake_cao, foreman_harness):
    """The queue's authoritative labels walk: enabled -> active -> done."""
    loop, queue, fake = foreman_harness(seed_issue_numbers=[1])
    _script_worker(fake_cao, loop.run_id)

    loop.run_pass()

    labels = fake.get_labels(1)
    assert "v3-work-done" in labels, f"expected 'v3-work-done' on issue 1, got {labels}"
    assert "v3-work" not in labels, f"expected 'v3-work' to be removed, got {labels}"

    state = queue.load_state("owner/repo#1")
    assert state is not None
    assert state.phase == "done"
    assert state.terminal_reason == "ci green"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scenario_1_repeated_runs_do_not_duplicate(fake_cao, foreman_harness, seed):
    """Running the foreman pass twice on the same issue must not
    duplicate branches or PRs. The foreman's claim should find nothing
    ready on the second pass because the issue is already ``done``."""
    loop, _queue, fake = foreman_harness(seed_issue_numbers=[seed])
    _script_worker(fake_cao, loop.run_id)

    first = loop.run_pass()
    second = loop.run_pass()

    assert len(first) == 1
    assert first[0].final_phase == "done"
    # The second pass has nothing to claim.
    assert second == []
    # No duplicate PR.
    open_prs_for_issue = [pr for pr in fake.list_open_prs() if pr.number == seed]
    assert len(open_prs_for_issue) <= 1, (
        f"second pass duplicated the PR for issue {seed}: {open_prs_for_issue}"
    )
