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
