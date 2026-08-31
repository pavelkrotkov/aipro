"""Tests for the V3 CI/PR gate (issue #55)."""

from __future__ import annotations

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.v3.ci_gate import CIPRGateImpl
from ai_pr_orchestrator.v3.config import CIPolicyConfig
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, GitHubPullRequestRef

SHA = "abc123"


def _pr() -> GitHubPullRequestRef:
    return GitHubPullRequestRef(owner="owner", repo="repo", number=5, head_sha=SHA)


def _issue() -> GitHubIssueRef:
    return GitHubIssueRef(owner="owner", repo="repo", number=1)


def _green_check(fake: FakeGitHubClient, name: str) -> None:
    fake.seed_check_run(SHA, name, "completed", "success")


def test_passes_when_all_checks_green():
    fake = FakeGitHubClient()
    _green_check(fake, "build")
    fake.seed_commit_status(SHA, "lint", "completed", "success")
    gate = CIPRGateImpl(fake, CIPolicyConfig(required_checks=["build"]))
    decision = gate.evaluate(_issue(), _pr())
    assert decision.passed
    assert decision.pending_checks == () and decision.failed_checks == ()


def test_failed_check_blocks():
    fake = FakeGitHubClient()
    fake.seed_check_run(SHA, "build", "completed", "failure")
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.failed_checks == ("build",)


def test_pending_check_blocks_and_cannot_pass():
    fake = FakeGitHubClient()
    fake.seed_check_run(SHA, "build", "in_progress", None)
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.pending_checks == ("build",)


def test_missing_required_check_fails_by_name():
    fake = FakeGitHubClient()
    _green_check(fake, "build")
    gate = CIPRGateImpl(fake, CIPolicyConfig(required_checks=["build", "e2e"]))
    decision = gate.evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.failed_checks == ("e2e",)


def test_no_checks_fails_when_green_required():
    fake = FakeGitHubClient()
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert "no checks" in decision.detail


def test_no_checks_passes_when_policy_disables_green_requirement():
    fake = FakeGitHubClient()
    cfg = CIPolicyConfig(require_green_ci_before_merge=False)
    decision = CIPRGateImpl(fake, cfg).evaluate(_issue(), _pr())
    assert decision.passed


def test_commit_status_failure_counts():
    fake = FakeGitHubClient()
    fake.seed_commit_status(SHA, "ci/state", "completed", "error")
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.failed_checks == ("ci/state",)


def test_stale_check_run_conclusion_is_failed():
    """A check-run conclusion of ``stale`` (no timely response) is a failure,
    not a pass: absence of a conclusion must never read as green."""
    fake = FakeGitHubClient()
    fake.seed_check_run(SHA, "build", "completed", "stale")
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.failed_checks == ("build",)


def test_no_checks_with_required_marks_them_pending_not_missing():
    """Right after a push a required check may simply not have reported yet;
    the gate classifies that initial absence as pending (in-flight), not a
    green pass and not a definitive failure."""
    fake = FakeGitHubClient()
    decision = CIPRGateImpl(fake, CIPolicyConfig(required_checks=["e2e"])).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.pending_checks == ("e2e",)
    assert decision.failed_checks == ()


def test_no_checks_no_required_still_falls_back_to_green_policy():
    """With no required_checks declared, the no-checks behaviour is driven by
    require_green_ci_before_merge as before."""
    fake = FakeGitHubClient()
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert "no checks" in decision.detail


def test_commit_statuses_classify_via_status_and_conclusion_fields():
    """Adapted Statuses-API contexts carry pending/completion in `status`
    (``in_progress``/``completed``) and the outcome in `conclusion`. The gate
    already classifies against those fields, so an in-flight context is pending
    and a failed one is failed."""
    fake = FakeGitHubClient()
    fake.seed_commit_status(SHA, "ci/lint", "in_progress", None)
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.pending_checks == ("ci/lint",)

    fake = FakeGitHubClient()
    fake.seed_commit_status(SHA, "ci/lint", "completed", "failure")
    decision = CIPRGateImpl(fake).evaluate(_issue(), _pr())
    assert not decision.passed
    assert decision.failed_checks == ("ci/lint",)
