"""Tests for ai_pr_orchestrator.v3.queue (issue #43)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3 import queue as q
from ai_pr_orchestrator.v3.config import GitHubQueueConfig
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, ReviewerFinding, WorkflowState
from ai_pr_orchestrator.v3.interfaces import StateConflictError


def _issue(number: int = 1) -> GitHubIssueRef:
    return GitHubIssueRef(owner="owner", repo="repo", number=number)


def _queue(fake: FakeGitHubClient, *, dry_run: bool = False, **cfg_kwargs) -> q.GitHubIssueQueue:
    cfg = GitHubQueueConfig(**cfg_kwargs)
    return q.GitHubIssueQueue(fake, "owner", "repo", cfg, host_id="host-A", dry_run=dry_run)


def _ready_fake(number: int = 1) -> FakeGitHubClient:
    fake = FakeGitHubClient()
    fake.seed_issue(number, labels=["v3-work"])
    return fake


# --- claim / competing claims ------------------------------------------------


def test_claim_success_creates_state_and_active_label():
    fake = _ready_fake()
    queue = _queue(fake)
    state = queue.claim(_issue(), "run-1", branch="b", worktree="/wt", pr_number=7)

    assert state.phase == "claiming"
    assert state.run_id == "run-1"
    # block persisted
    loaded = queue.load_state(state.work_item_id)
    assert loaded is not None and loaded.run_id == "run-1"
    # labels transitioned
    assert "v3-work-active" in fake.get_labels(1)
    assert "v3-work" not in fake.get_labels(1)
    # claim attribution carried
    claim = q.claim_from_state(loaded)
    assert claim.host_id == "host-A"
    assert claim.branch == "b" and claim.worktree == "/wt" and claim.pr_number == 7


def test_competing_claims_second_fails():
    fake = _ready_fake()
    queue = _queue(fake)
    queue.claim(_issue(), "run-1")
    with pytest.raises(q.ClaimConflictError):
        queue.claim(_issue(), "run-2")


def test_claim_round_trips_through_domain_types():
    """WorkItem + WorkflowState reconstruct solely from GitHub (no local cache)."""
    fake = _ready_fake()
    queue = _queue(fake)
    queue.claim(_issue(), "run-1")

    # Simulate a restart: a fresh queue built only on the same GitHub client.
    restarted = _queue(fake)
    wi = restarted.load_work_item(_issue())
    assert wi.id == "owner/repo#1"
    state = restarted.load_state(wi.id)
    assert state is not None and state.run_id == "run-1"
    assert q.claim_from_state(state).host_id == "host-A"


# --- lifecycle + label idempotency -------------------------------------------


def test_transition_to_review_sets_review_label_and_clears_active():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    s2 = queue.mark_review(_issue(), s)
    assert s2.phase == "reviewing"
    assert "v3-work-review" in fake.get_labels(1)
    assert "v3-work-active" not in fake.get_labels(1)


def test_complete_sets_done_label_and_terminal_reason():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    done = queue.complete(_issue(), s, reason="all ACs met")
    assert done.phase == "done"
    assert done.terminal_reason == "all ACs met"
    assert "v3-work-done" in fake.get_labels(1)
    assert "v3-work-active" not in fake.get_labels(1)


def test_label_transitions_are_idempotent():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    # call transition to the same target twice
    queue.transition(_issue(), s, "coding")
    s2 = queue.load_state("owner/repo#1")
    assert s2 is not None
    queue.transition(_issue(), s2, "coding")
    assert fake.get_labels(1).count("v3-work-active") == 1


def test_fail_sets_error_label():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    failed = queue.fail(_issue(), s, reason="reviewer crash")
    assert failed.phase == "failed"
    assert "v3-work-error" in fake.get_labels(1)


# --- optimistic concurrency ---------------------------------------------------


def test_update_with_stale_precondition_raises_conflict():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    # Another writer advances it first.
    queue.transition(_issue(), s, "coding")
    # Our handle `s` is stale -> save must fail.
    with pytest.raises(StateConflictError):
        queue.save_state(s, expected_updated_at=s.updated_at)


def test_create_only_raises_when_state_exists():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    with pytest.raises(StateConflictError):
        queue.save_state(s, expected_updated_at=None)


def test_update_with_matching_precondition_succeeds():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    new = s.transition("coding")
    queue.save_state(new, expected_updated_at=s.updated_at)
    loaded = queue.load_state("owner/repo#1")
    assert loaded is not None and loaded.phase == "coding"


# --- heartbeat + stale lease ---------------------------------------------------


def test_heartbeat_refreshes_lease():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    later = queue.heartbeat(s, now=datetime(2026, 1, 1, 0, 2, tzinfo=UTC))
    c = q.claim_from_state(later)
    assert c.heartbeat_at == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
    assert c.lease_expires_at == datetime(2026, 1, 1, 0, 2, tzinfo=UTC) + timedelta(seconds=900)


def test_stale_claim_cannot_be_reclaimed_while_active():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(q.ClaimConflictError):
        queue.reclaim_expired(_issue(), s, "run-2", now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC))


def test_reclaim_expired_after_lease():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    # 10 minutes later leases default 900s=15m have not lapsed yet -> bump past
    stale_now = datetime(2026, 1, 1, 0, 16, tzinfo=UTC)
    assert queue.is_claim_stale(s, stale_now)
    s2 = queue.reclaim_expired(_issue(), s, "run-2", now=stale_now)
    assert s2.run_id == "run-2"
    assert q.claim_from_state(s2).host_id == "host-A"
    assert "v3-work-active" in fake.get_labels(1)


# --- malformed / round trip / human text ---------------------------------------


def test_malformed_state_comment_raises():
    fake = FakeGitHubClient()
    fake.seed_issue(1, ["v3-work"])
    fake.seed_comment(1, "plain human comment, no block")
    queue = _queue(fake)
    assert queue.load_state("owner/repo#1") is None  # no block -> no state

    fake.seed_comment(1, "<!-- aipro-v3-state:start -->\nnot json\n<!-- aipro-v3-state:end -->")
    with pytest.raises(q.MalformedStateError):
        queue.load_state("owner/repo#1")


def test_human_text_preserved_around_block():
    fake = _ready_fake()
    queue = _queue(fake)
    queue.claim(_issue(), "run-1")
    # Simulate a human adding prose to the state comment, then advance state.
    comment = queue._find_state_comment(1)
    assert comment is not None
    human = "Reviewer wanted X. This is important context."
    with_prose = f"{human}\n\n{comment.body}"
    fake.edit_comment(comment.id, with_prose)

    s = queue.load_state("owner/repo#1")
    assert s is not None
    queue.transition(_issue(), s, "coding")
    updated = queue._find_state_comment(1)
    assert updated is not None
    assert human in updated.body  # human text survived the edit
    assert "aipro-v3-state:start" in updated.body


# --- dry-run -------------------------------------------------------------------


def test_dry_run_does_not_mutate_github():
    fake = FakeGitHubClient()
    fake.seed_issue(1, ["v3-work"])
    queue = _queue(fake, dry_run=True)
    s = queue.claim(_issue(), "run-1")
    # Returned state is present but nothing hit the transport.
    assert queue._find_state_comment(1) is None
    assert fake.get_labels(1) == ["v3-work"]  # labels untouched
    assert s.run_id == "run-1"


# --- compaction ----------------------------------------------------------------


def test_compaction_drops_oldest_findings():
    fake = _ready_fake()
    cfg = GitHubQueueConfig(max_state_block_chars=1200)
    queue = q.GitHubIssueQueue(fake, "owner", "repo", cfg, host_id="host-A")
    s = queue.claim(_issue(), "run-1")
    findings = [
        ReviewerFinding(
            id=f"f{i}",
            lane="breaker",
            body="x" * 80,
            severity="major",
            run_id="run-1",
            round_id="r1",
        )
        for i in range(30)
    ]
    big = WorkflowState(
        work_item_id=s.work_item_id,
        run_id="run-1",
        phase="reviewing",
        findings=findings,
    )
    queue.save_state(big, expected_updated_at=s.updated_at)
    loaded = queue.load_state("owner/repo#1")
    assert loaded is not None and loaded.phase == "reviewing"
    assert len(loaded.findings) < 30  # compacted
    assert loaded.findings[-1].id == "f29"  # most-recent retained
    assert loaded.findings[0].id != "f0"  # oldest dropped


def test_compaction_raises_if_still_too_large():
    fake = _ready_fake()
    cfg = GitHubQueueConfig(max_state_block_chars=200)
    queue = q.GitHubIssueQueue(fake, "owner", "repo", cfg, host_id="host-A")
    # Claim with the default (roomy) config so the initial state persists.
    plain = _queue(fake)
    s = plain.claim(_issue(), "run-1")
    big = WorkflowState(
        work_item_id=s.work_item_id,
        run_id="run-1",
        phase="coding",
        findings=[
            ReviewerFinding(
                id="f", lane="breaker", body="y" * 400, severity="major", run_id="r", round_id="x"
            )
        ],
    )
    with pytest.raises(q.StateBlockTooLargeError):
        queue.save_state(big, expected_updated_at=s.updated_at)


def test_list_ready_returns_enabled_issues():
    fake = FakeGitHubClient()
    fake.seed_issue(1, ["v3-work"])
    fake.seed_issue(2, ["v3-work"])
    fake.seed_issue(3, ["other"])
    queue = _queue(fake)
    ready = queue.list_ready()
    assert [r.number for r in ready] == [1, 2]
