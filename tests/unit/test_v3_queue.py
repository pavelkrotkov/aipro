"""Tests for ai_pr_orchestrator.v3.queue (issue #43)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3 import queue as q
from ai_pr_orchestrator.v3.config import GitHubQueueConfig
from ai_pr_orchestrator.v3.domain import (
    DomainError,
    GitHubIssueRef,
    ReviewerFinding,
    WorkflowState,
)
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

    fake.seed_comment(1, "<!-- v3-runtime-state:start -->\nnot json\n<!-- v3-runtime-state:end -->")
    with pytest.raises(q.MalformedStateError):
        queue.load_state("owner/repo#1")


def test_lone_state_marker_is_conflict_not_unclaimed():
    fake = FakeGitHubClient()
    fake.seed_issue(1, ["v3-work"])
    # A bare start marker (e.g. interrupted migration) is malformed authoritative
    # state, not "unclaimed": a create-only claim must refuse to add a second.
    fake.seed_comment(1, "<!-- v3-runtime-state:start -->")
    queue = _queue(fake)
    with pytest.raises(q.ClaimConflictError):
        queue.claim(_issue(), "run-1")


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
    assert "v3-runtime-state:start" in updated.body


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
            thread_id=f"T{i}",  # threaded -> recoverable from GitHub, droppable
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


# --- round-1 remediation regression tests -----------------------------------


def test_create_only_claim_arbitrates_duplicate_comments():
    """Concurrent claimants both POST; the earliest comment wins, other is deleted."""
    fake = _ready_fake()
    queue = _queue(fake)
    # Simulate the race: a concurrent claimant already posted a state comment with an
    # earlier id than a second POST would get.
    forged = WorkflowState(work_item_id="owner/repo#1", run_id="other-run", phase="claiming")
    i1 = fake.seed_comment(1, f"other\n\n{queue._embed_block(forged.to_dict())}")
    mine = WorkflowState(work_item_id="owner/repo#1", run_id="my-run", phase="claiming")
    created = fake.seed_comment(1, queue._embed_block(mine.to_dict()), comment_id=99)
    # Arbitrate: the lower-id comment (concurrent, earlier) wins and raises for us.
    with pytest.raises(q.ClaimConflictError):
        queue._settle_after_create(_issue(), created)
    survivors = [c for c in fake.get_pr_comments(1) if queue._block_re.search(c.body)]
    assert len(survivors) == 1
    assert survivors[0].id == i1.id
    assert fake.get_comment(99) is None  # our duplicate was deleted


def test_concurrent_post_write_clobber_is_detected():
    class _StealingFake(FakeGitHubClient):
        """edit_comment replaces the body AND then a concurrent writer wins."""

        def edit_comment(self, comment_id, body):
            mc = self._comments[comment_id]
            ours = mc.body
            super().edit_comment(comment_id, body)
            # Concurrent writer immediately overwrites the same comment.
            stolen = WorkflowState(
                work_item_id="owner/repo#1",
                run_id="intruder",
                phase="coding",
            )
            super().edit_comment(comment_id, f"{ours}\n\nbookkeeping")
            self._comments[comment_id].body = queue._embed_block(stolen.to_dict())
            return self._comments[comment_id].to_model()

    fake = _StealingFake()
    fake.seed_issue(1, ["v3-work"])
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    with pytest.raises(StateConflictError):
        # The intruder clobbered our write between edit and verify.
        queue.save_state(s.transition("coding"), expected_updated_at=s.updated_at)


def test_heartbeat_rejects_expired_lease():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    expired = datetime(2026, 1, 1, 0, 16, tzinfo=UTC)  # > 900s lease
    assert queue.is_claim_stale(s, expired)
    with pytest.raises(q.ClaimConflictError):
        queue.heartbeat(s, now=expired)


def test_compaction_keeps_unthreaded_findings():
    fake = _ready_fake()
    cfg = GitHubQueueConfig(max_state_block_chars=1500)
    queue = q.GitHubIssueQueue(fake, "owner", "repo", cfg, host_id="host-A")
    s = queue.claim(_issue(), "run-1")
    threaded = [
        ReviewerFinding(
            id=f"t{i}",
            lane="breaker",
            body="x" * 80,
            severity="major",
            run_id="run-1",
            round_id="r1",
            thread_id=f"T{i}",
        )
        for i in range(10)
    ]
    unthreaded = [
        ReviewerFinding(
            id="u1",
            lane="breaker",
            body="y" * 200,
            severity="blocker",
            run_id="run-1",
            round_id="r1",
        )
    ]
    big = WorkflowState(
        work_item_id=s.work_item_id,
        run_id="run-1",
        phase="reviewing",
        findings=unthreaded + threaded,
    )
    queue.save_state(big, expected_updated_at=s.updated_at)
    loaded = queue.load_state("owner/repo#1")
    assert loaded is not None
    ids = {f.id for f in loaded.findings}
    assert "u1" in ids  # unthreaded always preserved
    assert len(ids) < 11  # some threaded dropped to fit


def test_marker_derived_from_config():
    fake = _ready_fake()
    cfg = GitHubQueueConfig(state_comment_marker="custom-marker")
    queue = q.GitHubIssueQueue(fake, "owner", "repo", cfg, host_id="h")
    queue.claim(_issue(), "run-1")
    comment = queue._find_state_comment(1)
    assert comment is not None and "custom-marker:start" in comment.body


def test_heartbeat_preserves_non_claim_extras():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    tagged = replace(
        s.transition("coding"),
        extras={**s.extras, "forward_compat_key": "keep-me"},
    )
    queue.save_state(tagged, expected_updated_at=s.updated_at)
    loaded = queue.load_state("owner/repo#1")
    assert loaded is not None
    hb = queue.heartbeat(loaded)
    assert hb.extras.get("forward_compat_key") == "keep-me"


def test_reclaim_preserves_workflow_history():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    findings = [
        ReviewerFinding(
            id="f1",
            lane="breaker",
            body="nope",
            severity="major",
            run_id="run-1",
            round_id="r1",
        )
    ]
    advanced = replace(
        s.transition("reviewing", round_id="r1"),
        findings=findings,
        extras={**s.extras, "custom": "ctx"},
    )
    queue.save_state(advanced, expected_updated_at=s.updated_at)
    stale_now = datetime(2026, 1, 1, 0, 16, tzinfo=UTC)
    assert queue.is_claim_stale(advanced, stale_now)
    reclaimed = queue.reclaim_expired(_issue(), advanced, "run-2", now=stale_now)
    assert reclaimed.run_id == "run-2"
    assert [f.id for f in reclaimed.findings] == ["f1"]  # findings preserved
    assert reclaimed.round_id == "r1"  # round preserved
    assert reclaimed.extras.get("custom") == "ctx"  # non-claim extras preserved
    assert "v3-work-active" in fake.get_labels(1)


def test_reclaim_refuses_terminal_states():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    done = queue.complete(_issue(), s, reason="merged")
    old = datetime(2026, 1, 1, 0, 17, tzinfo=UTC)
    with pytest.raises(q.NoActiveClaimError):
        queue.reclaim_expired(_issue(), done, "run-2", now=old)


def test_queued_work_is_claimable():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    requeued = queue.transition(_issue(), s, "queued")
    assert requeued.phase == "queued"
    assert "v3-work" in fake.get_labels(1)  # back to enabled/ready
    # A second foreman can now claim it through the normal path.
    s2 = queue.claim(_issue(), "run-2")
    assert s2.run_id == "run-2"
    assert "v3-work-active" in fake.get_labels(1)
    assert "v3-work" not in fake.get_labels(1)


def test_naive_now_is_normalized_to_utc():
    fake = _ready_fake()
    queue = _queue(fake)
    naive = datetime(2026, 1, 1, 12, 0)  # naive
    s = queue.claim(_issue(), "run-1", now=naive)
    assert s.updated_at.tzinfo is not None
    # An immediate transition using the returned state must not false-conflict.
    s2 = queue.transition(_issue(), s, "coding")
    assert s2.phase == "coding"


def test_cross_repo_work_item_id_rejected():
    fake = _ready_fake()
    queue = _queue(fake)
    with pytest.raises(DomainError):
        queue.load_state("otherowner/otherrepo#1")
    other = WorkflowState(work_item_id="otherowner/otherrepo#1", run_id="r", phase="claiming")
    with pytest.raises(DomainError):
        queue.save_state(other, expected_updated_at=None)


def test_label_repair_after_partial_failure():
    gate = _LabelGate()
    queue = _queue(gate)
    s = queue.claim(_issue(), "run-1")
    gate.fail_labels = True
    with pytest.raises(q.LabelSyncError):
        queue.transition(_issue(), s, "reviewing")
    # Re-enable the transport, then repair labels to match the committed phase.
    gate.fail_labels = False
    queue.repair_labels(_issue())
    assert "v3-work-review" in gate.get_labels(1)


class _LabelGate(FakeGitHubClient):
    """Delegates to a fresh FakeGitHubClient but can force label writes to fail."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_labels = False

    def add_label(self, issue_number: int, label: str) -> list[dict[str, Any]]:
        if self.fail_labels:
            raise RuntimeError("label API down")
        return super().add_label(issue_number, label)

    def remove_label(self, issue_number: int, label: str) -> None:
        if self.fail_labels:
            raise RuntimeError("label API down")
        return super().remove_label(issue_number, label)


def test_custom_lease_seconds_respected():
    fake = _ready_fake()
    queue = _queue(fake, lease_seconds=60)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    assert s.extras["lease_expires_at"] == "2026-01-01T00:01:00+00:00"


def test_empty_lifecycle_label_rejected():
    from ai_pr_orchestrator.v3.config import V3Config, V3ConfigError

    with pytest.raises(V3ConfigError):
        V3Config(github_queue=GitHubQueueConfig(enabled_label="")).validate()
    with pytest.raises(V3ConfigError):
        V3Config(github_queue=GitHubQueueConfig(state_comment_marker="")).validate()


# --- round-2 remediation regression tests -----------------------------------


def test_claim_arbitration_fails_when_created_comment_disappears():
    class _VanishingFake(FakeGitHubClient):
        """Mimics a concurrent claimant's arbitration deleting our comment."""

        def get_pr_comments(self, issue_number):
            # After our POST lands, the rescan sees nothing (our comment was deleted).
            return []

    fake = _VanishingFake()
    queue = _queue(fake)
    with pytest.raises(q.ClaimConflictError):
        queue.claim(_issue(), "run-1")


def test_claim_requeues_preserve_non_claim_extras():
    fake = _ready_fake()
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    tagged = replace(
        s.transition("queued"),
        extras={**s.extras, "adapter_field": "survive-me"},
    )
    queue.save_state(tagged, expected_updated_at=s.updated_at)
    s2 = queue.claim(_issue(), "run-2")
    assert s2.extras.get("adapter_field") == "survive-me"


def test_claim_label_failure_raises_label_sync_error():
    gate = _LabelGate()
    queue = _queue(gate)
    gate.fail_labels = True
    with pytest.raises(q.LabelSyncError):
        queue.claim(_issue(), "run-1")


def test_reclaim_label_failure_raises_label_sync_error():
    gate = _LabelGate()
    queue = _queue(gate)
    s = queue.claim(_issue(), "run-1", now=datetime(2026, 1, 1, tzinfo=UTC))
    # Park the issue under the review label so reclaim must add/remove labels.
    gate.add_label(1, "v3-work-review")
    stale_now = datetime(2026, 1, 1, 0, 16, tzinfo=UTC)
    gate.fail_labels = True
    with pytest.raises(q.LabelSyncError):
        queue.reclaim_expired(_issue(), s, "run-2", now=stale_now)


def test_transition_rejects_split_identity():
    fake = _ready_fake()
    fake.seed_issue(2, [])
    queue = _queue(fake)
    s = queue.claim(_issue(), "run-1")
    other_issue = GitHubIssueRef(owner="owner", repo="repo", number=2)
    with pytest.raises(DomainError):
        queue.transition(other_issue, s, "coding")
    with pytest.raises(DomainError):
        queue.repair_labels(other_issue, s)


def test_heartbeat_rejected_from_non_owner_host():
    fake = _ready_fake()
    owner_queue = _queue(fake)  # host-A
    s = owner_queue.claim(_issue(), "run-1")
    other_queue = q.GitHubIssueQueue(fake, "owner", "repo", GitHubQueueConfig(), host_id="host-B")
    with pytest.raises(q.ClaimConflictError):
        other_queue.heartbeat(s)
    # The owner can still heartbeat.
    s2 = owner_queue.heartbeat(s)
    assert s2.extras["heartbeat_at"] is not None
